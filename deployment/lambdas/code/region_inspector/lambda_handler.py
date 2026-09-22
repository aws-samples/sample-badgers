"""Region Inspector Lambda (inspect_region_tool).

Re-reads the specific page regions a specialist flagged as uncertain. Called by the
orchestrator after the specialists for a page and before correlation. For each flagged
region it crops the region from the page image, enlarges it to the vision model's working
size, blind-reads it, compares the reading to the specialist's candidates, caps confidence
when the crop was enlarged past the point of new detail, and writes one JSON result to S3.

Pipeline per region: crop (region_inspector.py) -> upload crop -> blind read
(blind_reader.py) -> compare + cap (comparison.py) -> assemble. See KNOWLEDGE.md §4.

Follows the full_text_specialist handler scaffolding: job IDs read before the try, S3/local
config, save to S3, mark_complete, fail-loud on missing OUTPUT_BUCKET.
"""

import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import boto3

from foundation.s3_result_saver import save_result_to_s3
from foundation.s3_config_loader import load_manifest_from_s3
from foundation.model_selection import max_tokens_from, parse_model_selection
from foundation.bedrock_client import BedrockClient
from foundation import job_state

import comparison
from blind_reader import load_prompts, read_region
from region_inspector import RegionRequest, inspect_regions

logger = logging.getLogger()
log_level = os.environ.get("LOGGING_LEVEL", "INFO").upper()
logger.setLevel(getattr(logging, log_level, logging.INFO))

# The blind read is a short transcription of one small crop; it does not need the
# specialists' large output budget.
BLIND_READ_MAX_TOKENS = 1024


def lambda_handler(
    event: Dict[str, Any], context: Any
) -> Dict[str, Any]:  # noqa: ARG001
    """Lambda handler for the Region Inspector."""
    job_id = ""
    subtask = ""
    try:
        body = json.loads(event["body"]) if "body" in event else event

        session_id = body.get("session_id", "no_session")
        specialist_name = os.environ.get("SPECIALIST_NAME", "region_inspector")
        audit_mode = bool(body.get("audit_mode", False))
        image_path = body.get("image_path")
        regions_in = body.get("regions") or []

        logger.info(
            "Region inspection for session %s: %d region(s)",
            session_id,
            len(regions_in),
        )

        job_id = body.get("job_id") or ""
        doc_id = body.get("doc_id") or ""
        subtask = job_state.subtask_id(specialist_name, image_path)
        job_state.mark_running(
            job_id,
            subtask,
            doc_id=doc_id,
            specialist=specialist_name,
            image_id=job_state.image_identifier(image_path),
            session_id=session_id,
        )

        if not regions_in:
            job_state.mark_failed(
                job_id, subtask, "regions must contain at least one region"
            )
            return _error_response("regions must contain at least one region")

        image_bytes = _get_image_data(body)
        image_uri = image_path or "image_data"

        requests = [_to_region_request(r) for r in regions_in]
        crop_results = inspect_regions(image_bytes, requests, image_uri=image_uri)

        # Only build the Bedrock client and load the model selection if at least one region
        # actually needs a read (all-error input should not require Bedrock).
        needs_read = any(cr.error is None for cr in crop_results)
        bedrock_client = None
        model_selection = None
        prompts = None
        if needs_read:
            config = _load_config_from_s3(specialist_name)
            model_selection = parse_model_selection(config)
            prompts = load_prompts(os.environ.get("CONFIG_BUCKET", ""))
            bedrock_client = BedrockClient(
                throttle_delay=float(os.environ.get("THROTTLE_DELAY", "1.0")),
                aws_region=os.environ.get("AWS_REGION", "us-west-2"),
            )

        output_bucket = os.environ.get("OUTPUT_BUCKET")
        if not output_bucket:
            raise RuntimeError(
                "OUTPUT_BUCKET is not configured; inspection crops and results cannot "
                "be persisted."
            )
        s3 = boto3.client("s3")

        regions_out: List[Dict[str, Any]] = []
        for crop in crop_results:
            regions_out.append(
                _process_region(
                    crop,
                    s3=s3,
                    output_bucket=output_bucket,
                    session_id=session_id,
                    bedrock_client=bedrock_client,
                    model_selection=model_selection,
                    prompts=prompts,
                )
            )

        result_doc = {
            "specialist": "region_inspector",
            "session_id": session_id,
            "page": _derive_page(body, image_path),
            "audit_mode": audit_mode,
            "regions": regions_out,
        }
        result = json.dumps(result_doc, indent=2)

        s3_uri = save_result_to_s3(
            result=result,
            specialist_name=specialist_name,
            output_bucket=output_bucket,
            session_id=session_id,
            image_path=image_path,
        )
        result = f"{result}\n<!-- S3_RESULT_URI: {s3_uri} -->"
        job_state.mark_complete(job_id, subtask, s3_uri)

        return {
            "statusCode": 200,
            "body": json.dumps(
                {"result": result, "success": True, "session_id": session_id}
            ),
        }

    except (
        Exception
    ) as e:  # noqa: BLE001 - mirror the sibling handlers' fail-loud contract
        logger.error("Error: %s", e, exc_info=True)
        job_state.mark_failed(job_id, subtask, str(e))
        return {
            "statusCode": 500,
            "body": json.dumps({"result": str(e), "success": False}),
        }


def _process_region(
    crop,
    *,
    s3: Any,
    output_bucket: str,
    session_id: str,
    bedrock_client: Any,
    model_selection: Any,
    prompts: Any = None,
) -> Dict[str, Any]:
    """Turn one crop result into the §4.3 region dict: upload the crop, blind-read it,
    compare to the candidates, and cap. A crop that failed validation is returned as-is with
    its error and no reading."""
    region: Dict[str, Any] = {
        "region_id": crop.region_id,
        "image_uri": crop.image_uri,
        "flagged_by": crop.flagged_by,
        "task": crop.task,
        "page_px_size": crop.page_px_size,
        "source_px_box": crop.source_px_box,
        "source_px_size": crop.source_px_size,
        "output_px_size": crop.output_px_size,
        "scale_factor": crop.scale_factor,
        "detail": crop.detail,
        "notes": crop.notes,
        "crop_uri": None,
        "reading": None,
        "per_character": [],
        "marks": [],
        "concern": crop.concern,
        "match": None,
        "confidence": None,
        "capped": False,
        "error": crop.error,
    }

    if crop.error is not None:
        return region

    # Persist the crop so a human (or the report) can see exactly what the model read.
    crop_key = f"{session_id}/region_inspector/crops/{crop.region_id}.png"
    s3.put_object(
        Bucket=output_bucket,
        Key=crop_key,
        Body=base64.b64decode(crop.png_base64),
        ContentType="image/png",
    )
    region["crop_uri"] = f"s3://{output_bucket}/{crop_key}"

    reading = read_region(
        bedrock_client,
        model_selection,
        crop.png_base64,
        crop.task,
        prompts=prompts,
        max_tokens=BLIND_READ_MAX_TOKENS,
    )
    region["reading"] = reading["reading"]
    region["per_character"] = reading["per_character"]
    region["marks"] = reading["marks"]
    if reading.get("notes"):
        region["notes"] = list(region["notes"]) + [reading["notes"]]

    candidates = comparison.extract_candidates(crop.concern)
    region["match"] = comparison.match_reading(reading["reading"], candidates)

    confidence = comparison.confidence_from_per_character(
        reading["per_character"], candidates
    )
    confidence, capped = comparison.apply_cap(
        confidence, crop.source_px_size, crop.scale_factor
    )
    region["confidence"] = confidence
    region["capped"] = capped
    return region


def _to_region_request(raw: Dict[str, Any]) -> RegionRequest:
    return RegionRequest(
        region_id=str(raw.get("region_id", "")),
        x1=float(raw["x1"]),
        y1=float(raw["y1"]),
        x2=float(raw["x2"]),
        y2=float(raw["y2"]),
        task=str(raw.get("task", "transcribe")),
        flagged_by=str(raw.get("flagged_by", "")),
        concern=str(raw.get("concern", "")),
    )


def _derive_page(body: Dict[str, Any], image_path: Any):
    """Best-effort page number: explicit page_number, else the page_NNN suffix of the image
    filename, else None."""
    if body.get("page_number") is not None:
        try:
            return int(body["page_number"])
        except (TypeError, ValueError):
            return body["page_number"]
    if image_path:
        match = re.search(r"page[_-]?(\d+)", str(image_path), flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _load_config_from_s3(specialist_name: str) -> Dict[str, Any]:
    config_bucket = os.environ.get("CONFIG_BUCKET", "")
    manifest = load_manifest_from_s3(config_bucket, specialist_name)
    config: Dict[str, Any] = manifest.get("specialist", manifest)
    return config


def _get_image_data(body: Dict[str, Any]) -> bytes:
    """Extract image bytes from the request (base64 or S3/local path). Mirrors the
    full_text_specialist handler, including .b64 pre-encoded page files."""
    if "image_data" in body:
        return base64.b64decode(body["image_data"])

    if "image_path" in body:
        image_path = body["image_path"]

        if image_path.startswith("s3://"):
            s3 = boto3.client("s3")
            parts = image_path.replace("s3://", "").split("/", 1)
            bucket, key = parts[0], parts[1]
            response = s3.get_object(Bucket=bucket, Key=key)
            data = response["Body"].read()
            if key.endswith(".b64"):
                logger.info("Loading pre-encoded base64 from %s", image_path)
                return base64.b64decode(data.decode("utf-8"))
            return bytes(data)

        file_path = Path("/var/task") / image_path
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {image_path}")
        with open(file_path, "rb") as f:
            return f.read()

    raise ValueError("Missing image_data or image_path")


def _error_response(message: str) -> Dict[str, Any]:
    return {
        "statusCode": 500,
        "body": json.dumps({"result": message, "success": False}),
    }
