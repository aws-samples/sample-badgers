"""
Image Enhancer Lambda Handler

Agentic image enhancement using Claude Sonnet 4.6 vision model with Strands Agents.
Container-based Lambda to handle OpenCV/NumPy/Strands dependencies.
"""

import json
import logging
import os
import tempfile
import base64
from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np
import boto3

from agentic_enhancer import EnhancementUtilities
from enhancement_tools import load_image, save_image, image_to_base64

logger = logging.getLogger()
log_level = os.environ.get("LOGGING_LEVEL", "INFO").upper()
logger.setLevel(getattr(logging, log_level, logging.INFO))


def lambda_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """Lambda handler for agentic image enhancement."""
    try:
        body = json.loads(event["body"]) if "body" in event else event

        image_source = body.get("image_path") or body.get("image_data")
        document_type = body.get("document_type", "auto")
        enhancement_level = body.get("enhancement_level", "moderate")
        session_id = body.get("session_id", "no_session")
        output_quality = int(body.get("output_quality", 85))
        skip_upscale = body.get("skip_upscale", True)

        if not image_source:
            return _error_response("Missing required: image_path or image_data")

        # Load image
        if image_source.startswith("s3://"):
            local_path = _download_from_s3(image_source)
            # Handle .b64 files (base64 text files stored in S3)
            if local_path.endswith(".b64"):
                with open(local_path, "r") as f:
                    b64_data = f.read().strip()
                local_path = _save_base64_image(b64_data)
            image = load_image(local_path)
        elif image_source.startswith("data:") or len(image_source) > 500:
            # Base64 encoded image
            local_path = _save_base64_image(image_source)
            image = load_image(local_path)
        else:
            image = load_image(image_source)

        # Optional upscale (backward compatibility)
        original_shape = image.shape
        if not skip_upscale:
            image = _upscale_image(image, target_min_dimension=2000, target_max_dimension=4000)

        # Map parameters to agentic config
        context_str = _map_document_type_to_context(document_type)
        max_iterations_override = _map_enhancement_level_to_iterations(enhancement_level)

        # Temporarily override MAX_ITERATIONS env var
        original_max_iterations = os.environ.get("MAX_ITERATIONS")
        os.environ["MAX_ITERATIONS"] = str(max_iterations_override)

        try:
            # Run agentic enhancement
            result = EnhancementUtilities.enhance(
                image_source=image,
                context=context_str,
                save_output=False  # We handle S3 upload here
            )
        finally:
            # Restore original MAX_ITERATIONS
            if original_max_iterations:
                os.environ["MAX_ITERATIONS"] = original_max_iterations

        # Get winner image
        winner_image = result["winner_image"]

        # Save to temp file
        fd, output_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        save_image(winner_image, output_path, quality=output_quality)

        # Upload to S3 or return base64
        output_bucket = os.environ.get("OUTPUT_BUCKET")
        if output_bucket:
            s3_uri = _upload_to_s3(output_path, output_bucket, session_id, image_source)
            base64_data = None
        else:
            s3_uri = None
            with open(output_path, "rb") as f:
                base64_data = base64.b64encode(f.read()).decode("utf-8")

        # Format response (backward compatible + new fields)
        response_data = {
            # Old compatibility fields
            "s3_output_uri": s3_uri,
            "enhanced_image_base64": base64_data,
            "operations_applied": _extract_operations_list(result["history"]),
            "original_shape": list(original_shape),
            "final_shape": list(winner_image.shape),

            # New agentic fields
            "winner": result["winner"],
            "iterations": len(result["history"]),
            "reasoning": result.get("final_comparison", {}).get("reasoning", ""),
            "history": result["history"],
        }

        # Clean up temp file
        try:
            os.unlink(output_path)
        except:
            pass

        return {
            "statusCode": 200,
            "body": json.dumps({"result": response_data, "success": True}),
        }

    except Exception as e:
        logger.error("Error: %s", e, exc_info=True)
        return _error_response(str(e))


def _map_document_type_to_context(doc_type: str) -> str:
    """Map document_type to LLM context string."""
    mapping = {
        "manuscript": "18th century handwritten manuscript",
        "annotated": "historical document with handwritten annotations",
        "sheet_music": "musical score with performance annotations",
        "diagram": "technical diagram or chart",
        "printed": "printed historical document",
        "mixed": "mixed media document with multiple content types",
        "auto": "",
    }
    return mapping.get(doc_type.lower(), "")


def _map_enhancement_level_to_iterations(level: str) -> int:
    """Map enhancement_level to MAX_ITERATIONS."""
    mapping = {
        "minimal": 1,
        "moderate": 2,
        "aggressive": 3,
    }
    return mapping.get(level.lower(), 2)


def _upscale_image(
    image: np.ndarray,
    target_min_dimension: int = 2000,
    target_max_dimension: int = 4000
) -> np.ndarray:
    """
    Pre-process upscaling (backward compatibility).
    Extracted from old historical_document_enhancer.py.
    """
    h, w = image.shape[:2]
    min_dim = min(h, w)
    max_dim = max(h, w)

    # Check if upscaling needed
    if min_dim >= target_min_dimension:
        logger.info(f"Upscale skipped (already {w}x{h})")
        return image

    # Calculate scale factor
    scale = target_min_dimension / min_dim

    # Limit maximum size
    if max_dim * scale > target_max_dimension:
        scale = target_max_dimension / max_dim

    new_w = int(w * scale)
    new_h = int(h * scale)

    # Use INTER_CUBIC for upscaling (better for documents)
    upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    logger.info(f"Upscaled from {w}x{h} to {new_w}x{new_h}")
    return upscaled


def _extract_operations_list(history: list) -> list:
    """Extract flat list of operation names from history."""
    ops = []
    for iteration in history:
        for op in iteration.get("operations", []):
            ops.append(op.get("operation", op.get("op", "unknown")))
    return list(set(ops))  # Deduplicate


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


def _error_response(message: str) -> Dict[str, Any]:
    """Return error response."""
    return {
        "statusCode": 500,
        "body": json.dumps({"result": message, "success": False}),
    }
