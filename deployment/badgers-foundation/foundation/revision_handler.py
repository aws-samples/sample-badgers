"""foundation/revision_handler.py — revision-mode interceptor for all specialists.

This module provides a single function that wraps any specialist's lambda_handler
to intercept revision_mode requests. The specialist's handler code is UNCHANGED —
this wrapper sits in front of it.

The wrapper checks TWO things before allowing a revision call:
1. The request has ``revision_mode: true``
2. The specialist's manifest has ``revision_eligible: true``

If either check fails, the request falls through to the normal handler.

Usage — in each specialist's lambda_handler.py, change the handler function from:

    def lambda_handler(event, context):
        ...

To:

    from foundation.revision_handler import with_revision_support

    def _original_handler(event, context):
        ... (unchanged original code)

    lambda_handler = with_revision_support(_original_handler)

OR — if you prefer minimal diff — just add an early return at the top:

    from foundation.revision_handler import try_revision

    def lambda_handler(event, context):
        revision_result = try_revision(event, context)
        if revision_result is not None:
            return revision_result
        ... (unchanged original code)

Both patterns require ZERO changes to the specialist's analysis logic, config
loading, or initialization. The manifest flag ``revision_eligible`` is the only
thing that controls whether a specialist participates.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

from foundation.s3_result_saver import save_result_to_s3
from foundation import job_state

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  PUBLIC API — two integration patterns, pick one
# ═══════════════════════════════════════════════════════════════════════════


def with_revision_support(
    original_handler: Callable,
) -> Callable:
    """Decorator-style wrapper. Returns a new handler that intercepts revision
    requests and falls through to the original for normal requests.

    Usage:
        def _original_handler(event, context): ...
        lambda_handler = with_revision_support(_original_handler)
    """

    def wrapped_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
        revision_result = try_revision(event, context)
        if revision_result is not None:
            return revision_result
        return original_handler(event, context)

    return wrapped_handler


def try_revision(
    event: Dict[str, Any], context: Any  # noqa: ARG001
) -> Optional[Dict[str, Any]]:
    """Check if this is a revision request for a revision-eligible specialist.

    Returns the revision response if handled, or None if the request should
    fall through to the normal handler.
    """
    body = json.loads(event["body"]) if "body" in event else event

    if not body.get("revision_mode", False):
        return None  # Not a revision request — fall through

    specialist_name = os.environ.get("SPECIALIST_NAME", "")

    # Gate: check manifest for revision_eligible
    if not _is_revision_eligible(specialist_name):
        logger.warning(
            "%s received revision_mode=true but is not revision_eligible; "
            "rejecting. Set revision_eligible=true in the manifest to enable.",
            specialist_name,
        )
        return {
            "statusCode": 400,
            "body": json.dumps({
                "result": (
                    f"{specialist_name} does not have revision_eligible=true "
                    "in its manifest"
                ),
                "success": False,
            }),
        }

    return _handle_revision(body, specialist_name)


# ═══════════════════════════════════════════════════════════════════════════
#  INTERNAL — revision execution
# ═══════════════════════════════════════════════════════════════════════════


def _is_revision_eligible(specialist_name: str) -> bool:
    """Check the specialist's manifest for revision_eligible=true."""
    config_bucket = os.environ.get("CONFIG_BUCKET", "")
    if not config_bucket or not specialist_name:
        return False

    try:
        from foundation.s3_config_loader import load_manifest_from_s3

        manifest = load_manifest_from_s3(config_bucket, specialist_name)
        specialist_config = manifest.get("specialist", manifest)
        return bool(specialist_config.get("revision_eligible", False))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not load manifest for %s to check revision_eligible: %s",
            specialist_name,
            exc,
        )
        return False


def _handle_revision(
    body: Dict[str, Any],
    specialist_name: str,
) -> Dict[str, Any]:
    """Execute the revision pass: initialize the specialist at full capability
    and re-read each cropped region."""
    revision_regions = body.get("revision_regions") or []
    session_id = body.get("session_id", "no_session")
    audit_mode = body.get("audit_mode", False)
    job_id = body.get("job_id") or ""
    doc_id = body.get("doc_id") or ""

    revision_name = f"{specialist_name}_revision"

    if not revision_regions:
        return _error(400, "revision_mode=true but no revision_regions provided")

    subtask = job_state.subtask_id(revision_name, body.get("image_path"))
    job_state.mark_running(
        job_id,
        subtask,
        doc_id=doc_id,
        specialist=revision_name,
        image_id="revision_crops",
        session_id=session_id,
    )

    logger.info(
        "Revision mode: re-reading %d region(s) for %s",
        len(revision_regions),
        specialist_name,
    )

    # Initialize the specialist at full capability — same config, same prompts,
    # same model as pass 1.
    specialist = _initialize_specialist_from_env(specialist_name)

    # Process each region — the specialist sees ONLY the crop bytes.
    region_results: List[Dict[str, Any]] = []
    for region_spec in revision_regions:
        region_id = region_spec.get("region_id", "unknown")
        concern = region_spec.get("concern", "")
        notes = region_spec.get("notes", [])

        try:
            crop_bytes = _get_crop_bytes(region_spec)

            # Full specialist analysis on the crop — same prompt chain as pass 1.
            result_text = specialist.analyze(
                crop_bytes, body.get("aws_profile"), audit_mode
            )

            region_results.append({
                "region_id": region_id,
                "pass": 2,
                "concern": concern,
                "inspector_notes": notes,
                "result": result_text,
                "error": None,
            })
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Revision failed for region %s: %s", region_id, exc, exc_info=True
            )
            region_results.append({
                "region_id": region_id,
                "pass": 2,
                "concern": concern,
                "inspector_notes": notes,
                "result": None,
                "error": str(exc),
            })

    # Persist the revision result.
    result_doc = {
        "specialist": specialist_name,
        "mode": "revision",
        "pass": 2,
        "session_id": session_id,
        "audit_mode": audit_mode,
        "region_results": region_results,
    }
    result = json.dumps(result_doc, indent=2)

    output_bucket = os.environ.get("OUTPUT_BUCKET")
    if not output_bucket:
        raise RuntimeError("OUTPUT_BUCKET is not configured.")

    s3_uri = save_result_to_s3(
        result=result,
        specialist_name=revision_name,
        output_bucket=output_bucket,
        session_id=session_id,
        image_path=body.get("image_path"),
    )
    result = f"{result}\n<!-- S3_RESULT_URI: {s3_uri} -->"
    job_state.mark_complete(job_id, subtask, s3_uri)

    return {
        "statusCode": 200,
        "body": json.dumps(
            {"result": result, "success": True, "session_id": session_id}
        ),
    }


def _initialize_specialist_from_env(specialist_name: str):
    """Initialize the specialist using environment config — same pattern every
    handler uses. This avoids needing the handler's _initialize_specialist fn."""
    from foundation.specialist_foundation import SpecialistFoundation
    from foundation.configuration_manager import ConfigurationManager
    from foundation.prompt_loader import PromptLoader
    from foundation.image_processor import ImageProcessor
    from foundation.bedrock_client import BedrockClient
    from foundation.message_chain_builder import MessageChainBuilder
    from foundation.response_processor import ResponseProcessor
    from foundation.s3_config_loader import load_manifest_from_s3

    config_bucket = os.environ.get("CONFIG_BUCKET", "")

    # Load config — same as every handler does
    if config_bucket and os.environ.get("AWS_EXECUTION_ENV"):
        manifest = load_manifest_from_s3(config_bucket, specialist_name)
        config = manifest.get("specialist", manifest)
        config_source = "s3"
    else:
        from pathlib import Path

        manifest_path = Path("/var/task/manifest.json")
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        config = manifest["specialist"]
        config["prompt_base_path"] = str(
            Path("/var/task") / config["prompt_base_path"]
        )
        config["examples_path"] = str(Path("/var/task") / config["examples_path"])
        config_source = "local"

    specialist = object.__new__(SpecialistFoundation)
    specialist.specialist_type = specialist_name
    specialist.s3_bucket = config_bucket if config_source == "s3" else None
    specialist.logger = logging.getLogger(f"foundation.{specialist_name}")
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

    specialist.config_manager = ConfigurationManager()

    if config_source == "s3":
        specialist.prompt_loader = PromptLoader(
            config_source="s3",
            s3_bucket=config_bucket,
            specialist_name=specialist_name,
        )
    else:
        specialist.prompt_loader = PromptLoader(config_source="local")

    specialist.image_processor = ImageProcessor()
    specialist.bedrock_client = BedrockClient()
    specialist.message_builder = MessageChainBuilder()
    specialist.response_processor = ResponseProcessor()
    specialist._configure_components()

    return specialist


def _get_crop_bytes(region_spec: Dict[str, Any]) -> bytes:
    """Get crop image bytes from either crop_data (base64) or crop_uri (S3)."""
    if region_spec.get("crop_data"):
        return base64.b64decode(region_spec["crop_data"])

    crop_uri = region_spec.get("crop_uri", "")
    if crop_uri.startswith("s3://"):
        import boto3

        s3 = boto3.client("s3")
        parts = crop_uri.replace("s3://", "").split("/", 1)
        bucket, key = parts[0], parts[1]
        response = s3.get_object(Bucket=bucket, Key=key)
        return response["Body"].read()

    raise ValueError(
        f"Region {region_spec.get('region_id')}: "
        "missing both crop_data and crop_uri"
    )


def _error(status: int, msg: str) -> Dict[str, Any]:
    return {
        "statusCode": status,
        "body": json.dumps({"result": msg, "success": False}),
    }
