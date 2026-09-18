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
from typing import Dict, Any, Optional

import cv2
import numpy as np
import boto3

from agentic_enhancer import EnhancementUtilities
from enhancement_tools import load_image, save_image
from foundation import job_state

logger = logging.getLogger()
log_level = os.environ.get("LOGGING_LEVEL", "INFO").upper()
logger.setLevel(getattr(logging, log_level, logging.INFO))

# S3 config bucket (shared with agentic_enhancer.py prompt loader)
CONFIG_BUCKET = os.environ.get("CONFIG_BUCKET")

# Module-level cache for document type contexts (loaded once per cold start)
_cached_doc_type_contexts: Optional[Dict[str, str]] = None

def lambda_handler(event: Dict[str, Any], _context) -> Dict[str, Any]:
    """Lambda handler for agentic image enhancement."""
    # Job-tracking identifiers are read from the request body inside the try,
    # but must exist out here so the except handler can mark the subtask failed
    # even if the failure happened before they were parsed.
    job_id = ""
    subtask = ""
    try:
        body = json.loads(event["body"]) if "body" in event else event

        image_source = body.get("image_path") or body.get("image_data")
        document_type = body.get("document_type", "auto")
        enhancement_level = body.get("enhancement_level", "moderate")
        session_id = body.get("session_id", "no_session")
        output_quality = int(body.get("output_quality", 85))
        skip_upscale = body.get("skip_upscale", True)

        # Job tracking (doc_id -> job_id -> subtask_id). All of these no-op when
        # JOBS_TABLE_NAME is unset, so untracked deployments are unaffected.
        # Identity comes from image_path only: image_data is an inline payload
        # and would make a useless (and enormous) sort key.
        specialist_name = os.environ.get("SPECIALIST_NAME", "image_enhancer")
        job_id = body.get("job_id") or ""
        doc_id = body.get("doc_id") or ""
        subtask = job_state.subtask_id(specialist_name, body.get("image_path"))
        job_state.mark_running(
            job_id,
            subtask,
            doc_id=doc_id,
            specialist=specialist_name,
            image_id=job_state.image_identifier(body.get("image_path")),
            session_id=session_id,
        )

        if not image_source:
            job_state.mark_failed(
                job_id, subtask, "Missing required: image_path or image_data"
            )
            return _error_response("Missing required: image_path or image_data")

        # Load image
        if image_source.startswith("s3://"):
            local_path = _download_from_s3(image_source)
            # Handle .b64 files (base64 text files stored in S3)
            if local_path.endswith(".b64"):
                with open(local_path, "r", encoding="utf-8") as f:
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
            image = _upscale_image(
                image, target_min_dimension=2000, target_max_dimension=4000
            )

        # Map parameters to agentic config
        context_str = _map_document_type_to_context(document_type)
        max_iterations_override = _map_enhancement_level_to_iterations(
            enhancement_level
        )

        # Temporarily override MAX_ITERATIONS env var
        original_max_iterations = os.environ.get("MAX_ITERATIONS")
        os.environ["MAX_ITERATIONS"] = str(max_iterations_override)

        try:
            # Run agentic enhancement
            result = EnhancementUtilities.enhance(
                image_source=image,
                context=context_str,
                save_output=False,  # We handle S3 upload here
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

        # Upload to S3 or return base64.
        #
        # Unlike the other specialists this one has a genuine inline delivery
        # mode: with no OUTPUT_BUCKET it returns the enhanced image in the
        # response body, so nothing is missing and the caller still gets the
        # deliverable. Deployed Lambdas always receive OUTPUT_BUCKET from
        # LambdaSpecialistStack, so the base64 branch is a local-run affordance.
        # An _upload_to_s3 failure is not caught here and so still fails the
        # invocation via the outer handler.
        output_bucket = os.environ.get("OUTPUT_BUCKET")
        if output_bucket:
            s3_uri = _upload_to_s3(output_path, output_bucket, session_id, image_source)
            base64_data = None
        else:
            s3_uri = None
            with open(output_path, "rb") as f:
                base64_data = base64.b64encode(f.read()).decode("utf-8")

        job_state.mark_complete(job_id, subtask, s3_uri or "")

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
        except Exception:
            pass

        return {
            "statusCode": 200,
            "body": json.dumps({"result": response_data, "success": True}),
        }

    except Exception as e:
        logger.error("Error: %s", e, exc_info=True)
        job_state.mark_failed(job_id, subtask, str(e))
        return _error_response(str(e))


_FALLBACK_DOC_TYPE_CONTEXTS: Dict[str, str] = {
        "manuscript": (
            "18th century handwritten manuscript on laid or wove paper. "
            "Expect iron gall ink with possible corrosion halos, variable stroke "
            "weight from quill or steel nib, and natural baseline drift. Paper may "
            "show age-related yellowing, foxing spots, and bleed-through from the "
            "verso. Prioritize ink legibility over background aesthetics — gentle "
            "contrast and white balance corrections are usually sufficient."
        ),
        "annotated": (
            "Historical document with handwritten annotations overlaid on printed "
            "or manuscript base text. Multiple ink layers may be present — original "
            "text in one color/weight and later annotations in another (pencil, "
            "colored ink, or ballpoint). Marginal notes, interlinear glosses, "
            "underlines, brackets, and correction marks are common. Enhancement "
            "must preserve both layers without merging or suppressing either. "
            "Avoid aggressive contrast that could erase light pencil annotations."
        ),
        "sheet_music": (
            "Musical score with performance annotations. Contains precise geometric "
            "elements (staff lines, note heads, stems, beams, slurs) alongside "
            "handwritten markings (fingerings, dynamics, phrasing, rehearsal notes). "
            "Staff lines must remain continuous and even — avoid denoise or sharpen "
            "settings that could break thin horizontal lines. Handwritten annotations "
            "are often in pencil and lighter than the printed score. Prefer contrast "
            "enhancement over sharpening to maintain fine line integrity."
        ),
        "diagram": (
            "Technical diagram or chart — may include engineering drawings, "
            "architectural plans, flowcharts, circuit schematics, or scientific "
            "figures. Contains precise geometric lines, arrowheads, labels, and "
            "possibly hatching or cross-hatching patterns. Line weight variation "
            "is intentional and must be preserved. Text labels may be small and "
            "dense. Avoid denoise that could erode fine lines or merge closely "
            "spaced parallel lines. Deskew is critical — even small rotation "
            "misalignment is visually obvious in geometric content."
        ),
        "printed": (
            "Printed historical document — letterpress, lithograph, or early "
            "typewritten text. Uniform letterforms with possible impression "
            "artifacts (ink spread, uneven inking, strike-through ghosting). "
            "Paper may be brittle, yellowed, or show acid migration staining. "
            "Printed text is inherently higher contrast than handwriting, so "
            "moderate enhancement is usually sufficient. Watch for show-through "
            "from double-sided printing — denoise or remove_stains can help "
            "suppress verso bleed without harming the primary text."
        ),
        "mixed": (
            "Mixed media document with multiple content types — may combine "
            "printed text, handwriting, photographs, stamps, seals, colored "
            "illustrations, or pasted-in elements on a single page. Each "
            "content region may need different enhancement treatment. Consider "
            "regional operations: sharpen for text areas, gentle contrast for "
            "photographic regions, and stain removal for background. Global "
            "operations should be conservative to avoid degrading any single "
            "content type — prefer targeted regional enhancement."
        ),
        "historical_manuscript": (
            "Historical manuscript with period handwriting (secretary hand, "
            "court hand, or early modern cursive), likely 15th–18th century. "
            "Written on parchment or rag paper with iron gall ink, carbon ink, "
            "or sepia. Expect significant age degradation: foxing, tide lines, "
            "bleed-through, ink fading, and possible mold staining. The writing "
            "system may include period abbreviations (tildes, superscript letters, "
            "sigla), ligatures, and non-standard letterforms. Parchment backgrounds "
            "are typically warm-toned (cream to brown). Modern researcher annotations "
            "may be present — colored highlights, pencil marks, or sticky-note "
            "residue overlaid on the original. Enhancement priority: maximize "
            "ink-to-background contrast without clipping faded strokes. A "
            "desaturate → levels or desaturate → contrast pipeline is usually "
            "most effective. Avoid threshold unless the ink is uniformly dark — "
            "historical inks vary in density and thresholding destroys stroke "
            "weight information critical for paleographic reading."
        ),
        "photograph": (
            "Photographic content — may be a historical photograph, daguerreotype, "
            "tintype, or modern print/scan. Contains continuous tonal gradations "
            "rather than sharp text edges. Enhancement should preserve tonal "
            "subtlety — avoid over-sharpening that introduces halo artifacts "
            "or aggressive contrast that clips highlight/shadow detail. For "
            "faded photographs, equalize_histogram or gentle levels adjustment "
            "can recover lost tonal range. Scratches, dust, and emulsion damage "
            "may be present — denoise at low intensity can help without destroying "
            "grain structure."
        ),
        "map": (
            "Cartographic document — historical or technical map with geographic "
            "features, boundary lines, place-name labels, compass roses, scale "
            "bars, and possibly hand-colored regions. Contains both fine line "
            "work and text at multiple sizes and orientations. Color information "
            "may be semantically meaningful (boundary colors, terrain shading) — "
            "do NOT desaturate unless specifically instructed. Fold lines, tears, "
            "and water damage are common in historical maps. Enhancement should "
            "prioritize label legibility while preserving geographic line work. "
            "Deskew carefully — maps may have intentional non-orthogonal framing."
        ),
        "legal": (
            "Legal or administrative document — contracts, deeds, court records, "
            "certificates, or government filings. May contain pre-printed form "
            "fields with handwritten entries, stamps, seals (embossed or wax), "
            "signatures, notary marks, and official letterhead. Multiple ink "
            "colors and writing implements on a single page are common. "
            "Enhancement must preserve all layers — faint stamps and light-ink "
            "entries are as important as bold signatures. Avoid aggressive "
            "contrast that could erase light form-field entries or background "
            "security patterns."
        ),
        "newspaper": (
            "Newspaper or periodical page — dense multi-column layout with "
            "varying font sizes (headlines, body, captions), halftone photographs, "
            "line illustrations, and advertisements. Paper is typically low-quality "
            "newsprint with significant yellowing and brittleness. Halftone dots "
            "may create moiré patterns when scanned — gentle denoise can help. "
            "Column boundaries and text alignment should guide deskew. Show-through "
            "from verso is common on thin newsprint — remove_stains or levels "
            "adjustment can suppress it."
        ),
        "auto": "",
}


def _load_doc_type_contexts() -> Dict[str, str]:
    """Load document type context map from S3, falling back to hardcoded default.

    Mirrors the S3-first pattern from agentic_enhancer._load_system_prompt():
    1. Try S3:  s3://{CONFIG_BUCKET}/config/document_type_contexts.json
    2. Fall back to _FALLBACK_DOC_TYPE_CONTEXTS on any failure
    3. Cache at module level so subsequent warm invocations skip the S3 call

    The S3 JSON file should be a flat object mapping document type keys
    (lowercase) to context description strings, e.g.:
        {"manuscript": "18th century handwritten ...", "auto": "", ...}
    """
    global _cached_doc_type_contexts
    if _cached_doc_type_contexts is not None:
        return _cached_doc_type_contexts

    if CONFIG_BUCKET:
        try:
            s3 = boto3.client("s3")
            key = "config/document_type_contexts.json"
            logger.info(
                "Loading document type contexts from s3://%s/%s", CONFIG_BUCKET, key
            )
            response = s3.get_object(Bucket=CONFIG_BUCKET, Key=key)
            raw = response["Body"].read().decode("utf-8")
            loaded = json.loads(raw)

            if isinstance(loaded, dict):
                logger.info(
                    "Loaded %d document type contexts from S3", len(loaded)
                )
                _cached_doc_type_contexts = loaded
                return _cached_doc_type_contexts
            else:
                logger.warning("S3 document_type_contexts.json is not a dict, using fallback")
        except Exception as e:
            logger.warning("Failed to load document type contexts from S3: %s. Using fallback.", e)
    else:
        logger.info("No CONFIG_BUCKET set, using fallback document type contexts")

    _cached_doc_type_contexts = _FALLBACK_DOC_TYPE_CONTEXTS
    return _cached_doc_type_contexts


def _map_document_type_to_context(doc_type: str) -> str:
    """Map document_type to LLM context string.

    Loads the mapping from S3 on first call (with hardcoded fallback),
    then returns the context for the requested document type.
    """
    contexts = _load_doc_type_contexts()
    return contexts.get(doc_type.lower(), "")


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
    target_max_dimension: int = 4000,
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
        logger.info("Upscale skipped (already %dx%d)", w, h)
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

    logger.info("Upscaled from %dx%d to %dx%d", w, h, new_w, new_h)
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

    s3.upload_file(
        local_path, bucket, output_key,
        ExtraArgs={"ContentType": "image/jpeg"},
    )
    return f"s3://{bucket}/{output_key}"


def _error_response(message: str) -> Dict[str, Any]:
    """Return error response."""
    return {
        "statusCode": 500,
        "body": json.dumps({"result": message, "success": False}),
    }
