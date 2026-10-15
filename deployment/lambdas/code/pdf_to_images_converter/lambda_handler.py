"""PDF to Images Converter Lambda with AgentCore Gateway session tracking."""

import json
import logging
import base64
import os
import uuid
from pathlib import Path
from foundation import job_state
from foundation.lambda_error_handler import (
    create_error_response,
    ValidationError,
    ResourceNotFoundError,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Output bucket from environment (set by Lambda stack)
OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET", "")
import subprocess
import tempfile


def lambda_handler(event, context):
    import os

    logger.info("/opt/etc/fonts exists: %s", os.path.exists("/opt/etc/fonts"))
    logger.info("/opt/share/fonts exists: %s", os.path.exists("/opt/share/fonts"))
    """Convert PDF to images and store base64 in S3 temp location."""
    result = subprocess.run(["/opt/bin/pdftoppm", "-v"], capture_output=True, text=True)
    logger.info("pdftoppm version: %s %s", result.stdout, result.stderr)

    import shutil

    logger.info("pdf2image will use pdftoppm at: %s", shutil.which("pdftoppm"))

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

        # Parse input - AgentCore Gateway passes parameters directly in event
        body = json.loads(event["body"]) if "body" in event else event

        # Extract session_id from AgentCore Runtime (use provided or generate new).
        # The fallback is resolved here rather than after conversion so the
        # session id is available to job tracking and to every log line below.
        session_id = body.get("session_id") or uuid.uuid4().hex[:12]

        document_path = body.get("document_path")
        if not document_path:
            raise ValidationError(
                message="Missing required parameter: document_path",
                details={"provided_keys": list(body.keys())},
            )

        # Job tracking (doc_id -> job_id -> subtask_id). All of these no-op when
        # JOBS_TABLE_NAME is unset, so untracked deployments are unaffected.
        # This converter fans one source document out into page images, so its
        # subtask is keyed off the source document rather than a page.
        specialist_name = os.environ.get("SPECIALIST_NAME", "pdf_to_images_converter")
        job_id = body.get("job_id") or ""
        doc_id = body.get("doc_id") or ""
        subtask = job_state.subtask_id(specialist_name, document_path)
        job_state.mark_running(
            job_id,
            subtask,
            doc_id=doc_id,
            specialist=specialist_name,
            image_id=job_state.image_identifier(document_path),
            session_id=session_id,
        )

        max_image_size_mb = body.get("max_image_size_mb", 4.0)
        dpi = body.get("dpi", 128)
        page_numbers = _parse_page_numbers(body.get("page_numbers"))

        # Fetch the source bytes, then route on content. A PDF is rasterized
        # page-by-page via poppler; an already-image upload is a single page and
        # only needs normalization (RGB, downscale, JPEG-compress) — pdf2image
        # never runs on it. Both branches emit the same page_NNN.b64 artifacts,
        # so everything downstream (classify, specialists, correlator, report)
        # is identical regardless of source type.
        document_data = _get_document_data(document_path)

        if b"%PDF-" in document_data[:1024]:
            rendered_pages = _convert_pdf_to_images(
                document_data, dpi, max_image_size_mb, page_numbers=page_numbers
            )
        else:
            if page_numbers and page_numbers != [1]:
                raise ValidationError(
                    message="An image input has exactly one page; page_numbers may only contain 1.",
                    details={"page_numbers": page_numbers},
                )
            rendered_pages = [(1, _convert_image_to_base64(document_data, max_image_size_mb))]

        # Store base64 in S3 temp location. _store_images_to_s3 raises when
        # OUTPUT_BUCKET is unset, so a missing artifact destination already
        # fails the invocation rather than returning an empty success.
        logger.info("Processing request for runtime session_id: %s", session_id)

        rendered_page_paths = _store_images_to_s3(rendered_pages, session_id)
        s3_paths = [page["image_path"] for page in rendered_page_paths]

        # The converter's artifact is the set of page images; record the first as
        # the representative key so the row points into the right prefix.
        job_state.mark_complete(job_id, subtask, s3_paths[0] if s3_paths else "")

        logger.info(
            "Converted source document to %d page image(s), stored in S3 temp/ (session: %s)",
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
                            "pages": rendered_page_paths,
                            "session_id": session_id,
                        }
                    ),
                    "success": True,
                    "session_id": session_id,
                }
            ),
        }

    except Exception as e:
        job_state.mark_failed(job_id, subtask, str(e))
        return create_error_response(e)


def _decrypt_pdf_if_needed(pdf_data: bytes) -> bytes:
    logger.info("Starting PDF decryption with qpdf")
    with (
        tempfile.NamedTemporaryFile(suffix=".pdf") as infile,
        tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as outfile,
    ):
        infile.write(pdf_data)
        infile.flush()
        logger.info("Input PDF size: %d bytes", len(pdf_data))
        result = subprocess.run(
            ["/opt/bin/qpdf", "--decrypt", infile.name, outfile.name],
            capture_output=True,
            text=True,
        )
        logger.info(
            "qpdf returncode: %d, stdout: %s, stderr: %s",
            result.returncode,
            result.stdout,
            result.stderr,
        )
        with open(outfile.name, "rb") as f:
            decrypted = f.read()
        logger.info("Decrypted PDF size: %d bytes", len(decrypted))
        return decrypted


def _get_document_data(document_path: str) -> bytes:
    """Get source document bytes (PDF or image) from S3 or a local path."""
    if document_path.startswith("s3://"):
        import boto3

        try:
            s3 = boto3.client("s3")
            parts = document_path.replace("s3://", "").split("/", 1)
            if len(parts) != 2:
                raise ValidationError(
                    message="Invalid S3 path format.",
                    details={
                        "document_path": document_path,
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
                message=f"Failed to download document from S3: {str(e)}",
                details={"document_path": document_path},
            )

    # Local file
    file_path = Path(document_path)
    if not file_path.exists():
        raise ResourceNotFoundError(
            message="Document file not found.",
            details={"document_path": document_path},
        )

    with open(file_path, "rb") as f:
        return f.read()


def _convert_image_to_base64(image_data: bytes, max_size_mb: float) -> str:
    """Normalize a single uploaded image into one base64-encoded JPEG page.

    Mirrors the per-page handling in _convert_pdf_to_images (RGB conversion +
    quality step-down) and adds a max-dimension downscale, since an uploaded
    image can be far larger than a rasterized PDF page. No pdf2image/poppler is
    involved. Only the first frame of a multi-frame image (animated GIF,
    multi-page TIFF) is used.
    """
    from PIL import Image
    import io

    max_dimension = 2048
    max_size_bytes = int(max_size_mb * 1024 * 1024)

    try:
        img = Image.open(io.BytesIO(image_data))
        img.load()  # force-decode the first frame so a corrupt file fails here
    except Exception as e:
        raise ValidationError(
            message="Unsupported or corrupt image file.",
            details={"error": str(e)},
        )

    # Flatten transparency onto white rather than letting convert("RGB") turn it
    # black — otherwise a transparent PNG renders as a black page.
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    if max(img.size) > max_dimension:
        img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

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
    logger.info("Normalized uploaded image: quality=%d, size=%d bytes", quality, size)
    return b64_str


def _parse_page_numbers(value) -> list[int] | None:
    """Validate an optional one-indexed, de-duplicated list of PDF pages."""
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise ValidationError(
            message="page_numbers must be a non-empty array of positive integers.",
            details={"page_numbers": value},
        )
    if any(not isinstance(page, int) or isinstance(page, bool) or page < 1 for page in value):
        raise ValidationError(
            message="page_numbers must contain only positive integers.",
            details={"page_numbers": value},
        )
    return sorted(set(value))


def _convert_pdf_to_images(
    pdf_data: bytes,
    dpi: int,
    max_size_mb: float,
    *,
    page_numbers: list[int] | None = None,
) -> list[tuple[int, str]]:
    """Convert all PDF pages, or only the explicitly requested pages, to JPEG base64."""
    from pdf2image import convert_from_bytes
    from PIL import Image
    import io

    pdf_data = _decrypt_pdf_if_needed(pdf_data)
    if page_numbers is None:
        pil_pages = list(enumerate(convert_from_bytes(pdf_data, dpi=dpi), start=1))
    else:
        pil_pages = []
        for page_number in page_numbers:
            try:
                rendered = convert_from_bytes(
                    pdf_data,
                    dpi=dpi,
                    first_page=page_number,
                    last_page=page_number,
                )
            except Exception as error:
                raise ValidationError(
                    message="Unable to render requested PDF page.",
                    details={"page_number": page_number, "error": str(error)},
                ) from error
            if len(rendered) != 1:
                raise ValidationError(
                    message="Requested PDF page does not exist.",
                    details={"page_number": page_number},
                )
            pil_pages.append((page_number, rendered[0]))

    rendered_pages: list[tuple[int, str]] = []
    max_size_bytes = int(max_size_mb * 1024 * 1024)

    for page_number, img in pil_pages:
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
        rendered_pages.append((page_number, b64_str))

        logger.info("Page %d: quality=%d, size=%d bytes", page_number, quality, size)

    return rendered_pages


def _store_images_to_s3(
    rendered_pages: list[tuple[int, str]], session_id: str
) -> list[dict[str, object]]:
    """Store page image artifacts, retaining the original PDF page number."""
    import boto3

    if not OUTPUT_BUCKET:
        raise ValidationError(
            message="OUTPUT_BUCKET_NAME environment variable not set",
            details={},
        )

    s3 = boto3.client("s3")
    pages = []

    for page_number, b64_str in rendered_pages:
        key = f"temp/{session_id}/page_{page_number:03d}.b64"

        # Store base64 string as text
        s3.put_object(
            Bucket=OUTPUT_BUCKET,
            Key=key,
            Body=b64_str.encode("utf-8"),
            ContentType="text/plain",
        )

        s3_path = f"s3://{OUTPUT_BUCKET}/{key}"
        pages.append({"page_number": page_number, "image_path": s3_path})
        logger.info("Stored page %d to %s", page_number, s3_path)

    return pages
