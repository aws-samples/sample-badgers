"""S3 Result Saver - Save analysis results to S3."""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import boto3

logger = logging.getLogger(__name__)

# Cache for manifest data
_manifest_cache: dict = {}


class S3ResultSaverError(Exception):
    """Error saving result to S3."""

    pass


def _get_output_extension(analyzer_name: str) -> str:
    """Get output extension from analyzer manifest.

    Args:
        analyzer_name: Name of the analyzer (e.g., 'full_text_analyzer')

    Returns:
        File extension ('xml' or 'json'), defaults to 'xml'
    """
    global _manifest_cache

    if analyzer_name in _manifest_cache:
        return _manifest_cache[analyzer_name]

    # Try to load from S3 manifest bucket or local file
    manifest_bucket = os.environ.get("MANIFEST_BUCKET")

    if manifest_bucket:
        try:
            s3 = boto3.client("s3")
            manifest_key = f"manifests/{analyzer_name}.json"
            response = s3.get_object(Bucket=manifest_bucket, Key=manifest_key)
            manifest = json.loads(response["Body"].read().decode("utf-8"))
            ext = manifest.get("analyzer", {}).get("output_extension", "xml")
            _manifest_cache[analyzer_name] = ext
            return ext
        except Exception as e:
            logger.warning("Could not load manifest for %s: %s", analyzer_name, e)

    # Default to xml
    _manifest_cache[analyzer_name] = "xml"
    return "xml"


def save_result_to_s3(
    result: str,
    analyzer_name: str,
    output_bucket: str,
    session_id: str,
    image_path: Optional[str] = None,
) -> str:
    """Save analysis result to S3.

    Args:
        result: The analysis result content to save
        analyzer_name: Name of the analyzer (used in S3 key path and to determine output format)
        output_bucket: S3 bucket name for output
        session_id: Session ID for organizing results
        image_path: Optional source image path for naming

    Returns:
        S3 URI of the saved result (s3://bucket/key)
    """
    import re

    s3 = boto3.client("s3")

    # Clean markdown artifacts as secondary safety measure
    result = re.sub(
        r"^```(?:xml|json|html|text)?\s*\n?", "", result, flags=re.IGNORECASE
    )
    result = re.sub(r"\n?```\s*$", "", result)
    result = result.strip()

    # Generate filename with timestamp
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")

    # Get source image identifier if available
    image_identifier = _extract_image_identifier(image_path)

    # Determine file extension from manifest config
    ext = _get_output_extension(analyzer_name)
    content_type = "application/json" if ext == "json" else "application/xml"
    filename = f"{analyzer_name}_{image_identifier}_{timestamp}.{ext}"
    s3_key = f"{session_id}/{analyzer_name}/{filename}"

    # Save to S3
    s3.put_object(
        Bucket=output_bucket,
        Key=s3_key,
        Body=result.encode("utf-8"),
        ContentType=content_type,
    )

    s3_uri = f"s3://{output_bucket}/{s3_key}"
    logger.info("Saved result to %s", s3_uri)

    return s3_uri


def _extract_image_identifier(image_path: Optional[str]) -> str:
    """Extract identifier from image path for naming."""
    if not image_path:
        return "unknown"

    if image_path.startswith("s3://"):
        # Extract filename from S3 path
        return image_path.split("/")[-1].rsplit(".", 1)[0]
    else:
        return Path(image_path).stem
