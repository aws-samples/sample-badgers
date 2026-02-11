"""
Image Enhancer Lambda Handler

Enhances historical document images for better vision LLM processing.
Container-based Lambda to handle OpenCV/NumPy dependencies.
"""

import json
import logging
import os
import tempfile
import base64
from pathlib import Path
from typing import Dict, Any, Optional

import boto3
from historical_document_enhancer import (
    HistoricalDocumentEnhancer,
    DocumentType,
    EnhancementLevel,
)

logger = logging.getLogger()
log_level = os.environ.get("LOGGING_LEVEL", "INFO").upper()
logger.setLevel(getattr(logging, log_level, logging.INFO))


def lambda_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """Lambda handler for image enhancement."""
    try:
        body = json.loads(event["body"]) if "body" in event else event

        image_source = body.get("image_path") or body.get("image_data")
        document_type = body.get("document_type", "auto")
        enhancement_level = body.get("enhancement_level", "moderate")
        session_id = body.get("session_id", "no_session")

        if not image_source:
            return _error_response("Missing required: image_path or image_data")

        # Load image
        if image_source.startswith("s3://"):
            local_path = _download_from_s3(image_source)
        elif image_source.startswith("data:") or len(image_source) > 500:
            # Base64 encoded image
            local_path = _save_base64_image(image_source)
        else:
            local_path = image_source

        # Map document type
        type_map = {
            "auto": DocumentType.UNKNOWN,
            "manuscript": DocumentType.MANUSCRIPT,
            "annotated": DocumentType.ANNOTATED,
            "sheet_music": DocumentType.SHEET_MUSIC,
            "diagram": DocumentType.TECHNICAL_DIAGRAM,
            "printed": DocumentType.PRINTED_HISTORICAL,
            "mixed": DocumentType.MIXED_MEDIA,
        }
        doc_type = type_map.get(document_type.lower(), DocumentType.UNKNOWN)

        # Map enhancement level
        level_map = {
            "minimal": EnhancementLevel.MINIMAL,
            "moderate": EnhancementLevel.MODERATE,
            "aggressive": EnhancementLevel.AGGRESSIVE,
        }
        level = level_map.get(enhancement_level.lower(), EnhancementLevel.MODERATE)

        # Skip upscale option - preserve original dimensions
        skip_upscale = body.get("skip_upscale", True)  # Default to skip

        # Enhance
        from historical_document_enhancer import EnhancementConfig

        config = EnhancementConfig()
        if skip_upscale:
            config.target_min_dimension = 0  # Disable upscaling
        enhancer = HistoricalDocumentEnhancer(config)
        result = enhancer.enhance(local_path, document_type=doc_type, level=level)

        # Save enhanced image as JPEG with quality setting
        output_quality = int(body.get("output_quality", 85))
        fd, output_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        _save_as_jpeg(result.enhanced_image, output_path, output_quality)

        # Upload to S3 if output bucket configured
        output_bucket = os.environ.get("OUTPUT_BUCKET")
        if output_bucket:
            s3_uri = _upload_to_s3(output_path, output_bucket, session_id, image_source)
            response_data = {
                "s3_output_uri": s3_uri,
                "operations_applied": result.operations_applied,
                "original_shape": result.original_shape,
                "final_shape": result.final_shape,
                "skew_angle": result.skew_angle,
            }
        else:
            # Return base64 encoded image
            with open(output_path, "rb") as f:
                enhanced_b64 = base64.b64encode(f.read()).decode("utf-8")
            response_data = {
                "enhanced_image_base64": enhanced_b64,
                "operations_applied": result.operations_applied,
                "original_shape": result.original_shape,
                "final_shape": result.final_shape,
                "skew_angle": result.skew_angle,
            }

        return {
            "statusCode": 200,
            "body": json.dumps({"result": response_data, "success": True}),
        }

    except Exception as e:
        logger.error("Error: %s", e, exc_info=True)
        return _error_response(str(e))


def _download_from_s3(s3_uri: str) -> str:
    """Download file from S3 to temp location."""
    s3 = boto3.client("s3")
    parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]

    ext = Path(key).suffix or ".png"
    fd, temp_path = tempfile.mkstemp(suffix=ext)
    os.close(fd)

    s3.download_file(bucket, key, temp_path)
    return temp_path


def _save_base64_image(data: str) -> str:
    """Save base64 encoded image to temp file."""
    if data.startswith("data:"):
        data = data.split(",", 1)[1]

    image_bytes = base64.b64decode(data)
    fd, temp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)

    with open(temp_path, "wb") as f:
        f.write(image_bytes)

    return temp_path


def _upload_to_s3(local_path: str, bucket: str, session_id: str, original: str) -> str:
    """Upload enhanced image to S3."""
    from datetime import datetime

    s3 = boto3.client("s3")

    original_name = Path(original).stem if "/" in original else "image"
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_key = f"{session_id}/enhanced/{original_name}_enhanced_{timestamp}.jpg"

    s3.upload_file(local_path, bucket, output_key)
    return f"s3://{bucket}/{output_key}"


def _save_as_jpeg(image: "np.ndarray", output_path: str, quality: int = 85) -> None:
    """Save image as JPEG with specified quality."""
    import cv2

    # Convert RGB to BGR for OpenCV
    if len(image.shape) == 3:
        save_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        save_img = image

    cv2.imwrite(output_path, save_img, [cv2.IMWRITE_JPEG_QUALITY, quality])


def _error_response(message: str) -> Dict[str, Any]:
    """Return error response."""
    return {
        "statusCode": 500,
        "body": json.dumps({"result": message, "success": False}),
    }
