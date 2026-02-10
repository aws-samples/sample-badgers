"""PDF to Images Converter Lambda with AgentCore Gateway session tracking."""

import json
import logging
import base64
import os
import uuid
from pathlib import Path
from foundation.lambda_error_handler import (
    create_error_response,
    ValidationError,
    ResourceNotFoundError,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Output bucket from environment (set by Lambda stack)
OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET", "")


def lambda_handler(event, context):
    """Convert PDF to images and store base64 in S3 temp location."""
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

        # Extract session_id from AgentCore Runtime (use provided or generate new)
        session_id = body.get("session_id")

        pdf_path = body.get("pdf_path")
        if not pdf_path:
            raise ValidationError(
                message="Missing required parameter: pdf_path",
                details={"provided_keys": list(body.keys())},
            )

        max_image_size_mb = body.get("max_image_size_mb", 4.0)
        dpi = body.get("dpi", 128)

        # Get PDF data
        pdf_data = _get_pdf_data(pdf_path)

        # Convert PDF to base64 images
        base64_images = _convert_pdf_to_images(pdf_data, dpi, max_image_size_mb)

        # Store base64 in S3 temp location (use provided session_id or generate new)
        if not session_id:
            session_id = uuid.uuid4().hex[:12]
        logger.info("Processing request for runtime session_id: %s", session_id)

        s3_paths = _store_images_to_s3(base64_images, session_id)

        logger.info(
            "Converted PDF to %d images, stored in S3 temp/ (session: %s)",
            len(s3_paths),
            session_id,
        )

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "result": json.dumps(
                        {
                            "images": s3_paths,
                            "page_count": len(s3_paths),
                            "session_id": session_id,
                        }
                    ),
                    "success": True,
                    "session_id": session_id,
                }
            ),
        }

    except Exception as e:
        return create_error_response(e)


def _get_pdf_data(pdf_path: str) -> bytes:
    """Get PDF data from S3 or local path."""
    if pdf_path.startswith("s3://"):
        import boto3

        try:
            s3 = boto3.client("s3")
            parts = pdf_path.replace("s3://", "").split("/", 1)
            if len(parts) != 2:
                raise ValidationError(
                    message="Invalid S3 path format.",
                    details={
                        "pdf_path": pdf_path,
                        "expected_format": "s3://bucket/key",
                    },
                )
            bucket, key = parts
            response = s3.get_object(Bucket=bucket, Key=key)
            return bytes(response["Body"].read())
        except ValidationError:
            raise
        except Exception as e:
            raise ResourceNotFoundError(
                message=f"Failed to download PDF from S3: {str(e)}",
                details={"pdf_path": pdf_path},
            )

    # Local file
    file_path = Path(pdf_path)
    if not file_path.exists():
        raise ResourceNotFoundError(
            message="PDF file not found.",
            details={"pdf_path": pdf_path},
        )

    with open(file_path, "rb") as f:
        return f.read()


def _convert_pdf_to_images(pdf_data: bytes, dpi: int, max_size_mb: float) -> list[str]:
    """Convert PDF bytes to base64-encoded images."""
    from pdf2image import convert_from_bytes
    from PIL import Image
    import io

    # Convert PDF to PIL images
    pil_images = convert_from_bytes(pdf_data, dpi=dpi)

    base64_images = []
    max_size_bytes = int(max_size_mb * 1024 * 1024)

    for idx, img in enumerate(pil_images):
        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Compress to meet size limit
        quality = 85
        while quality > 20:
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality, optimize=True)
            size = buffer.tell()

            if size <= max_size_bytes:
                break

            quality -= 10

        buffer.seek(0)
        b64_str = base64.b64encode(buffer.read()).decode("utf-8")
        base64_images.append(b64_str)

        logger.info("Page %d: quality=%d, size=%d bytes", idx + 1, quality, size)

    return base64_images


def _store_images_to_s3(base64_images: list[str], session_id: str) -> list[str]:
    """Store base64 images to S3 temp location, return S3 paths."""
    import boto3

    if not OUTPUT_BUCKET:
        raise ValidationError(
            message="OUTPUT_BUCKET_NAME environment variable not set",
            details={},
        )

    s3 = boto3.client("s3")
    s3_paths = []

    for idx, b64_str in enumerate(base64_images):
        key = f"temp/{session_id}/page_{idx + 1:03d}.b64"

        # Store base64 string as text
        s3.put_object(
            Bucket=OUTPUT_BUCKET,
            Key=key,
            Body=b64_str.encode("utf-8"),
            ContentType="text/plain",
        )

        s3_path = f"s3://{OUTPUT_BUCKET}/{key}"
        s3_paths.append(s3_path)
        logger.info("Stored page %d to %s", idx + 1, s3_path)

    return s3_paths
