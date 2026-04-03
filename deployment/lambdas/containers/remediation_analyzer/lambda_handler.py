"""
PDF Accessibility Tagger — Lambda Handler
==========================================
Wraps the final_form tagging pipeline for Lambda container deployment.

Pipeline: S3 download → parse correlation XML → resolve bboxes →
          wrap content stream → build structure tree → save → S3 upload

Event schema matches the existing remediation_analyzer handler for
drop-in compatibility.
"""

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import fitz
from pikepdf import Pdf

from utils.pdf_accessibility_models import TagRegion, VALID_TAGS
from utils.spine_parser import parse_correlation_xml

from pdf_syntax_repair import repair_pdf as syntax_repair_pdf, RepairResult

from utils import (
    map_tag,
    resolve_text_bboxes,
    resolve_figure_bboxes,
    wrap_content_stream_fixed,
    build_structure_tree,
)

logger = logging.getLogger()
log_level = os.environ.get("LOGGING_LEVEL", "INFO").upper()
logger.setLevel(getattr(logging, log_level, logging.INFO))


def lambda_handler(event, context):
    """Lambda entry point for PDF accessibility tagging."""
    try:
        body = json.loads(event["body"]) if "body" in event else event

        session_id = body.get("session_id", "no_session")
        pdf_path = body.get("pdf_path")
        title = body.get("title", "Accessible Document")
        lang = body.get("lang", "en-US")
        correlation_uri = body.get("correlation_uri")

        logger.info("Session: %s, PDF: %s", session_id, pdf_path)

        if not pdf_path:
            return _error_response("Missing required parameter: pdf_path")
        if not correlation_uri:
            return _error_response("Missing required parameter: correlation_uri")

        local_pdf = _download_from_s3(pdf_path)
        local_xml = _download_from_s3(correlation_uri)

        result = process_pdf(
            pdf_path=local_pdf,
            xml_path=local_xml,
            title=title,
            lang=lang,
            session_id=session_id,
        )

        # Upload outputs to S3 if configured
        output_bucket = os.environ.get("OUTPUT_BUCKET")
        analyzer_name = os.environ.get("ANALYZER_NAME", "remediation_analyzer")

        if output_bucket and result.get("output_pdf"):
            s3_uri = _upload_to_s3(
                local_path=result["output_pdf"],
                bucket=output_bucket,
                analyzer_name=analyzer_name,
                session_id=session_id,
                original_key=pdf_path,
            )
            result["s3_output_uri"] = s3_uri
            logger.info("Uploaded tagged PDF to: %s", s3_uri)

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "result": result,
                    "success": True,
                    "session_id": session_id,
                }
            ),
        }

    except Exception as e:
        logger.error("Error: %s", e, exc_info=True)
        return _error_response(str(e))


def process_pdf(
    pdf_path: str,
    xml_path: str,
    title: str = "Accessible Document",
    lang: str = "en-US",
    session_id: str = "",
) -> Dict[str, Any]:
    """Full tagging pipeline: XML parse → bbox resolve → content stream → structure tree.

    Returns dict with output_pdf path and diagnostics.
    """
    work_dir = Path(tempfile.mkdtemp(prefix="pdf_tagger_"))

    # --- Syntax repair pass (fix corrupt xref, streams, etc.) ---
    repair_result: RepairResult | None = None
    if os.environ.get("ENABLE_SYNTAX_REPAIR", "true").lower() == "true":
        try:
            repair_result = syntax_repair_pdf(pdf_path, work_dir=work_dir)
            if repair_result.any_repair_applied:
                logger.info(
                    "Syntax repair applied: %s → %s (%+d bytes)",
                    pdf_path,
                    repair_result.output_path,
                    repair_result.size_delta,
                )
                pdf_path = repair_result.output_path
            else:
                logger.info("Syntax repair: no changes needed")
        except Exception as e:
            logger.error("Syntax repair failed: %s", e, exc_info=True)
            raise RuntimeError(
                f"PDF syntax repair failed and the file could not be made processable. "
                f"Accessibility tagging was not attempted. "
                f"Repair error: {e}"
            ) from e

    # 1. Parse correlation XML
    with open(xml_path, encoding="utf-8") as f:
        xml_content = f.read()

    tree, page_elems = parse_correlation_xml(xml_content)
    logger.info(
        "Correlation: tree=%s, %d page(s)",
        "yes" if tree else "no",
        len(page_elems),
    )

    # Get correlation elements (handle page key mismatch)
    # For single-page correlation XMLs, all elements share one key
    all_corr_elems = []
    for pg_elems in page_elems.values():
        all_corr_elems.extend(pg_elems)

    text_elems = [e for e in all_corr_elems if e["type"] != "figure"]
    fig_elems = [e for e in all_corr_elems if e["type"] == "figure"]

    # 2. Open PDF with both fitz (for text search) and pikepdf (for writing)
    fitz_doc = fitz.open(pdf_path)
    pdf = Pdf.open(pdf_path)
    num_pages = len(fitz_doc)

    merged_mcid_map: Dict[int, TagRegion] = {}
    mcid_counter = 0

    for page_idx in range(num_pages):
        fitz_page = fitz_doc[page_idx]

        # Resolve bboxes for this page
        resolved_text = resolve_text_bboxes(fitz_page, text_elems)
        resolved_figs = resolve_figure_bboxes(fitz_page, fig_elems)
        resolved = resolved_text + resolved_figs

        matched = [
            e
            for e in resolved
            if e["source"] not in ("fallback_stacked", "fallback_figure")
        ]
        if not matched:
            logger.info("Page %d: no matches, skipping", page_idx + 1)
            continue

        logger.info(
            "Page %d: %d resolved (%d matched, %d fallback)",
            page_idx + 1,
            len(resolved),
            len(matched),
            len(resolved) - len(matched),
        )

        # Build TagRegion objects
        regions = _build_regions(resolved, fitz_page, page_idx)
        regions = {page_idx: regions.get(page_idx, [])}

        # Wrap content stream with BDC/EMC
        new_content, mcid_map, mcid_counter = wrap_content_stream_fixed(
            pdf,
            fitz_doc,
            regions,
            page_num=page_idx,
            mcid_start=mcid_counter,
        )

        if new_content and mcid_map:
            pdf.pages[page_idx]["/Contents"] = pdf.make_stream(new_content)
            merged_mcid_map.update(mcid_map)
            logger.info(
                "Page %d: %d MCIDs, %d bytes content stream",
                page_idx + 1,
                len(mcid_map),
                len(new_content),
            )

    # 3. Build structure tree ONCE with all merged MCIDs
    total_mcids = len(merged_mcid_map)
    total_elems = 0
    if merged_mcid_map:
        result_info = build_structure_tree(
            pdf,
            merged_mcid_map,
            page_num=0,
            structure_element_tree=tree,
        )
        if isinstance(result_info, int):
            total_elems = result_info
        else:
            total_elems = len(result_info)

    # 4. Set metadata
    try:
        with pdf.open_metadata() as meta:
            meta["dc:title"] = title
            meta["dc:language"] = [lang]
            meta["pdfuaid:part"] = "1"
    except Exception as e:
        logger.warning("Metadata write failed: %s", e)

    # 5. Save
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = str(work_dir / f"tagged_{ts}.pdf")
    pdf.save(output_path)
    pdf.close()
    fitz_doc.close()

    logger.info(
        "Tagged PDF saved: %s (%d MCIDs, %d StructElems, %d pages)",
        output_path,
        total_mcids,
        total_elems,
        num_pages,
    )

    return {
        "output_pdf": output_path,
        "pages_processed": num_pages,
        "total_mcids": total_mcids,
        "total_struct_elems": total_elems,
        "syntax_repair": {
            "applied": repair_result.any_repair_applied if repair_result else False,
            "pass1_ok": repair_result.pass1_ok if repair_result else False,
            "pass2_ok": repair_result.pass2_ok if repair_result else False,
            "size_delta": repair_result.size_delta if repair_result else 0,
            "logs": repair_result.logs if repair_result else [],
        },
    }


# ============================================================================
# Helpers
# ============================================================================


def _build_regions(
    resolved_elements: List[Dict],
    fitz_page,
    page_idx: int,
) -> Dict[int, List[TagRegion]]:
    """Convert resolved bbox dicts into TagRegion objects keyed by page index."""
    w, h = fitz_page.rect.width, fitz_page.rect.height
    regions: Dict[int, List[TagRegion]] = {}

    for elem in resolved_elements:
        bbox = elem["bbox"]
        x0 = bbox["x0"] * w
        x1 = bbox["x1"] * w
        y0 = (1 - bbox["y1"]) * h
        y1 = (1 - bbox["y0"]) * h

        tag = elem.get("tag") or map_tag(elem.get("type", "P"))
        if tag not in VALID_TAGS:
            tag = "P"

        region = TagRegion(
            tag=tag,
            bbox=(x0, y0, x1, y1),
            alt_text=elem.get("alt_text", ""),
            text_content=elem.get("content", ""),
            order=elem.get("order", 0),
            page=page_idx,
            element_id=elem.get("id", ""),
            source=elem.get("source", ""),
            enrichment_title=elem.get("enrichment_title") or None,
            enrichment_description=elem.get("enrichment_description") or None,
            enrichment_actual_text=elem.get("enrichment_actual_text") or None,
            enrichment_tags=elem.get("enrichment_tags") or None,
            enrichment_related=elem.get("enrichment_related") or None,
        )
        regions.setdefault(page_idx, []).append(region)

    return regions


def _download_from_s3(s3_uri: str) -> str:
    """Download file from S3 to temp location. Pass through local paths."""
    import boto3

    if not s3_uri.startswith("s3://"):
        return s3_uri

    s3 = boto3.client("s3")
    parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]

    ext = Path(key).suffix or ".pdf"
    fd, temp_path = tempfile.mkstemp(suffix=ext)
    os.close(fd)

    s3.download_file(bucket, key, temp_path)
    return temp_path


def _upload_to_s3(
    local_path: str,
    bucket: str,
    analyzer_name: str,
    session_id: str,
    original_key: str,
) -> str:
    """Upload file to S3 output bucket."""
    import boto3

    s3 = boto3.client("s3")

    original_name = Path(original_key).stem
    ext = Path(local_path).suffix or Path(original_key).suffix or ".pdf"
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_key = f"{session_id}/{analyzer_name}/{original_name}_{timestamp}{ext}"

    s3.upload_file(local_path, bucket, output_key)
    return f"s3://{bucket}/{output_key}"


def _error_response(message: str) -> Dict[str, Any]:
    """Return error response."""
    return {
        "statusCode": 500,
        "body": json.dumps({"result": message, "success": False}),
    }
