"""S3 Result Saver - Save analysis results to S3."""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import boto3

logger = logging.getLogger(__name__)

# Cache for manifest data
_manifest_cache: dict = {}


class S3ResultSaverError(Exception):
    """Error saving result to S3."""

    pass


def _get_output_extension(specialist_name: str) -> str:
    """Get output extension from specialist manifest.

    Args:
        specialist_name: Name of the specialist (e.g., 'full_text_specialist')

    Returns:
        File extension ('xml' or 'json'), defaults to 'xml'
    """
    global _manifest_cache

    if specialist_name in _manifest_cache:
        return _manifest_cache[specialist_name]

    # Try to load from S3 manifest bucket or local file
    manifest_bucket = os.environ.get("MANIFEST_BUCKET") or os.environ.get("CONFIG_BUCKET")

    if manifest_bucket:
        try:
            s3 = boto3.client("s3")
            from foundation.s3_config_loader import config_key

            manifest_key = config_key(f"manifests/{specialist_name}.json")
            response = s3.get_object(Bucket=manifest_bucket, Key=manifest_key)
            manifest = json.loads(response["Body"].read().decode("utf-8"))
            ext = manifest.get("specialist", {}).get("output_extension", "xml")
            _manifest_cache[specialist_name] = ext
            return ext
        except Exception as e:
            logger.warning("Could not load manifest for %s: %s", specialist_name, e)

    # Default to xml
    _manifest_cache[specialist_name] = "xml"
    return "xml"


def save_result_to_s3(
    result: str,
    specialist_name: str,
    output_bucket: str,
    session_id: str,
    image_path: Optional[str] = None,
    job_id: str = "",
) -> str:
    """Save analysis result to S3.

    Args:
        result: The analysis result content to save
        specialist_name: Name of the specialist (used in S3 key path and to determine output format)
        output_bucket: S3 bucket name for output
        session_id: Session ID for organizing results
        image_path: Optional source image path for naming
        job_id: Durable operation ID when this is runner-managed

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
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")

    # Get source image identifier if available
    image_identifier = _extract_image_identifier(image_path)

    # Determine file extension from manifest config
    ext = _get_output_extension(specialist_name)
    content_type = "application/json" if ext == "json" else "application/xml"
    filename = f"{specialist_name}_{image_identifier}_{timestamp}.{ext}"
    s3_key = f"{session_id}/{specialist_name}/{filename}"
    # A durable workflow never accepts an output key from a caller. When its
    # server-generated operation ID is present, derive the result prefix from
    # the operation's owner in DynamoDB. Legacy direct specialist invocation
    # retains the historical session-based layout.
    if job_id.startswith("op_"):
        try:
            from foundation import operation_state

            record = operation_state.get_operation(job_id)
            if not record or not record.get("owner_sub"):
                raise S3ResultSaverError(
                    "Runner-managed result has no durable operation owner"
                )
            s3_key = (
                f"{operation_state.operation_prefix(str(record['owner_sub']), job_id)}"
                f"/specialists/{specialist_name}/{filename}"
            )
        except S3ResultSaverError:
            raise
        except Exception as error:
            raise S3ResultSaverError(
                "Could not derive the private operation artifact prefix"
            ) from error

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
