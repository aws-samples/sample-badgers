"""Generate a document-level BADGERS HTML report from canonical page spines."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any

import boto3

from foundation import job_state
from renderer import render_report

logger = logging.getLogger()
logger.setLevel(
    getattr(logging, os.environ.get("LOGGING_LEVEL", "INFO").upper(), logging.INFO)
)

_S3_URI = re.compile(r"^s3://([a-z0-9][a-z0-9.-]{1,61}[a-z0-9])/([^?#]+)$")
_PAGE_NUMBER = re.compile(r"^[1-9][0-9]{0,5}$")
_MAX_PAGES = 50
_MAX_XML_BYTES = 5 * 1024 * 1024
_MAX_IMAGE_OBJECT_BYTES = 8 * 1024 * 1024
_MAX_AGGREGATE_BYTES = 150 * 1024 * 1024


def lambda_handler(
    event: dict[str, Any], context: Any
) -> dict[str, Any]:  # noqa: ARG001
    """Build durable page artifacts, a report manifest, and offline HTML."""
    job_id = ""
    subtask = ""
    session_id = "no_session"
    try:
        body = json.loads(event["body"]) if "body" in event else event
        source_document_path = str(body.get("source_document_path") or "")
        document_title = str(body.get("document_title") or "").strip()
        expected_page_count = int(body.get("expected_page_count") or 0)
        pages = body.get("pages") or []
        session_id = str(body.get("session_id") or "no_session")
        job_id = str(body.get("job_id") or "")
        doc_id = str(body.get("doc_id") or "")
        user_id = str(body.get("user_id") or "local")
        user_name = str(body.get("user_name") or "local")
        specialist_name = os.environ.get("SPECIALIST_NAME", "html_report_specialist")

        if not source_document_path.startswith("s3://"):
            raise ValueError("source_document_path must be an S3 URI")
        if not job_id or not doc_id:
            raise ValueError("job_id and doc_id are required for report ownership")
        if not user_id:
            raise ValueError("user_id is required for report ownership")
        if not isinstance(pages, list) or not pages:
            raise ValueError("pages must contain at least one correlated page")
        if expected_page_count < 1 or expected_page_count > _MAX_PAGES:
            raise ValueError(f"expected_page_count must be between 1 and {_MAX_PAGES}")
        if len(pages) != expected_page_count:
            raise ValueError(
                "pages must contain exactly one correlated spine for every PDF page"
            )
        if len(pages) > _MAX_PAGES:
            raise ValueError(f"reports support at most {_MAX_PAGES} pages")

        report_id = _safe_id(job_id or session_id, "report")
        owner_key = _owner_key(user_id)
        report_prefix = f"reports/{owner_key}/{report_id}"
        subtask = job_state.subtask_id(specialist_name, source_document_path)
        job_state.mark_running(
            job_id,
            subtask,
            doc_id=doc_id,
            specialist=specialist_name,
            image_id=job_state.image_identifier(source_document_path),
            session_id=session_id,
        )

        output_bucket = os.environ.get("OUTPUT_BUCKET", "")
        if not output_bucket:
            raise RuntimeError("OUTPUT_BUCKET is not configured")
        s3 = boto3.client("s3")
        job_records = job_state.get_job_records(job_id)
        if not job_records:
            raise ValueError("No job records found for report generation")
        job_row = next(
            (
                record
                for record in job_records
                if record.get("subtask_id") == "orchestrator"
            ),
            {},
        )
        if job_row.get("doc_id") != doc_id:
            raise ValueError("Report document does not match the analysis job")
        if job_row.get("owner_sub") != user_id:
            raise ValueError("Report user does not own the analysis job")
        correlation_records = {
            str(record.get("result_s3_key") or ""): record
            for record in job_records
            if record.get("specialist") == "correlation_specialist"
            and record.get("status") == "COMPLETE"
        }
        normalized_pages = _validate_pages(pages)
        report_pages: list[dict[str, Any]] = []
        aggregate_bytes = 0

        for page in normalized_pages:
            page_number = page["page_number"]
            correlation_uri = page["correlation_specialist_uri"]
            image_uri = page["source_image_path"]
            correlation_record = correlation_records.get(correlation_uri)
            if not correlation_record:
                raise ValueError(
                    f"Page {page_number} correlation artifact does not belong to this job"
                )
            if correlation_record.get("image_identifier") != job_state.image_identifier(
                image_uri
            ):
                raise ValueError(
                    f"Page {page_number} image does not match the job correlation record"
                )
            correlation_xml = _get_text(
                s3,
                correlation_uri,
                output_bucket,
                max_bytes=_MAX_XML_BYTES,
            )
            parsed = _parse_spine(correlation_xml)
            if parsed["source_image"] != image_uri:
                raise ValueError(
                    f"Page {page_number} image does not match its correlation artifact"
                )
            image_base64, image_bytes = _get_image(s3, image_uri, output_bucket)
            aggregate_bytes += len(correlation_xml.encode("utf-8")) + len(image_base64)
            if aggregate_bytes > _MAX_AGGREGATE_BYTES:
                raise ValueError("report artifacts exceed the 150 MiB aggregate limit")

            page_token = _safe_id(page_number, str(len(report_pages) + 1))
            image_key = f"{report_prefix}/pages/page-{page_token}.jpg"
            spine_key = f"{report_prefix}/pages/page-{page_token}.xml"
            s3.put_object(
                Bucket=output_bucket,
                Key=image_key,
                Body=image_bytes,
                ContentType="image/jpeg",
                CacheControl="private, max-age=3600",
            )
            s3.put_object(
                Bucket=output_bucket,
                Key=spine_key,
                Body=correlation_xml.encode("utf-8"),
                ContentType="application/xml; charset=utf-8",
                CacheControl="private, max-age=3600",
            )
            audit = _page_audit(
                job_records,
                image_uri,
                correlation_uri,
                parsed["specialists"],
            )
            report_pages.append(
                {
                    "page_number": page_number,
                    "summary": parsed["summary"],
                    "specialists": parsed["specialists"],
                    "elements": parsed["elements"],
                    "audit": audit,
                    "image_key": image_key,
                    "spine_key": spine_key,
                    "correlation_specialist_uri": correlation_uri,
                    "source_image_path": image_uri,
                    "image_data": image_base64,
                    "xml": correlation_xml,
                }
            )

        created_at = datetime.now(timezone.utc).isoformat()
        specialists = sorted(
            {
                specialist.get("name", "")
                for page in report_pages
                for specialist in page["specialists"]
                if specialist.get("name")
            }
        )
        element_count = sum(len(page["elements"]) for page in report_pages)
        complete_count = (
            sum(1 for record in job_records if record.get("status") == "COMPLETE") + 1
        )
        failed_count = sum(
            1 for record in job_records if record.get("status") == "FAILED"
        )
        title = document_title or source_document_path.rstrip("/").rsplit("/", 1)[-1]
        summary = (
            f"BADGERS analyzed {len(report_pages)} pages using {len(specialists)} "
            f"unique specialists and produced {element_count} structured elements."
        )
        report_model = {
            "schema_version": "1.0",
            "report_id": report_id,
            "owner_sub": user_id,
            "user_name": user_name,
            "title": title,
            "summary": summary,
            "source_document_path": source_document_path,
            "doc_id": doc_id,
            "job_id": job_id,
            "session_id": session_id,
            "created_at": created_at,
            "page_count": len(report_pages),
            "specialist_count": len(specialists),
            "specialists": specialists,
            "element_count": element_count,
            "invocation_count": len(
                [r for r in job_records if r.get("subtask_id") != "orchestrator"]
            ),
            "complete_count": complete_count,
            "failed_count": failed_count,
            "pages": report_pages,
        }
        html_report = render_report(report_model)
        html_key = f"{report_prefix}/report.html"
        manifest_key = f"{report_prefix}/manifest.json"
        manifest = _manifest(report_model, html_key, manifest_key)
        s3.put_object(
            Bucket=output_bucket,
            Key=html_key,
            Body=html_report.encode("utf-8"),
            ContentType="text/html; charset=utf-8",
            ContentDisposition=f'attachment; filename="{report_id}.html"',
        )
        s3.put_object(
            Bucket=output_bucket,
            Key=manifest_key,
            Body=json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
            ContentType="application/json; charset=utf-8",
            CacheControl="no-store",
        )
        html_uri = f"s3://{output_bucket}/{html_key}"
        manifest_uri = f"s3://{output_bucket}/{manifest_key}"
        job_state.mark_complete(job_id, subtask, manifest_uri)
        # Point the job-level row at this report so the UI can enumerate a user's
        # reports from the owner-index GSI instead of listing the output bucket.
        # Written only after both artifacts are durable: a pointer to a half
        # written report would surface in the listing and then 404 on open.
        job_state.set_report(
            job_id,
            report_id=report_id,
            title=title,
            created_at=created_at,
            page_count=len(report_pages),
        )
        return _response(
            200,
            html_report_uri=html_uri,
            report_manifest_uri=manifest_uri,
            summary=summary,
            page_count=len(report_pages),
            success=True,
            session_id=session_id,
        )
    except Exception as error:
        logger.error("HTML report generation failed: %s", error, exc_info=True)
        job_state.mark_failed(job_id, subtask, str(error))
        return _response(
            500,
            html_report_uri=None,
            report_manifest_uri=None,
            summary=f"Error: {error}",
            page_count=0,
            success=False,
            session_id=session_id,
        )


def _validate_pages(pages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, page in enumerate(pages, start=1):
        if not isinstance(page, dict):
            raise ValueError(f"Page entry {index} must be an object")
        page_number = str(page.get("page_number") or "").strip()
        correlation_uri = str(page.get("correlation_specialist_uri") or "").strip()
        image_uri = str(page.get("source_image_path") or "").strip()
        if not _PAGE_NUMBER.fullmatch(page_number):
            raise ValueError(
                f"Page entry {index} page_number must be a positive integer"
            )
        if not correlation_uri or not image_uri:
            raise ValueError(f"Page entry {index} is missing required artifact URIs")
        if page_number in seen:
            raise ValueError(f"Duplicate page number: {page_number}")
        seen.add(page_number)
        normalized.append(
            {
                "page_number": page_number,
                "correlation_specialist_uri": correlation_uri,
                "source_image_path": image_uri,
            }
        )
    return normalized


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    match = _S3_URI.fullmatch(uri)
    if not match:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return match.group(1), match.group(2)


def _get_text(s3: Any, uri: str, expected_bucket: str, *, max_bytes: int) -> str:
    bucket, key = _parse_s3_uri(uri)
    if bucket != expected_bucket:
        raise ValueError("Report inputs must come from the configured output bucket")
    response = s3.get_object(Bucket=bucket, Key=key)
    if int(response.get("ContentLength") or 0) > max_bytes:
        raise ValueError(f"Report input is larger than {max_bytes} bytes: {uri}")
    body = bytes(response["Body"].read(max_bytes + 1))
    if len(body) > max_bytes:
        raise ValueError(f"Report input is larger than {max_bytes} bytes: {uri}")
    return body.decode("utf-8")


def _get_image(s3: Any, uri: str, expected_bucket: str) -> tuple[str, bytes]:
    bucket, key = _parse_s3_uri(uri)
    if bucket != expected_bucket:
        raise ValueError("Report inputs must come from the configured output bucket")
    response = s3.get_object(Bucket=bucket, Key=key)
    if int(response.get("ContentLength") or 0) > _MAX_IMAGE_OBJECT_BYTES:
        raise ValueError(f"Report image exceeds 8 MiB: {uri}")
    body = bytes(response["Body"].read(_MAX_IMAGE_OBJECT_BYTES + 1))
    if len(body) > _MAX_IMAGE_OBJECT_BYTES:
        raise ValueError(f"Report image exceeds 8 MiB: {uri}")
    if key.lower().endswith(".b64"):
        encoded = body.decode("utf-8").strip()
        try:
            return encoded, base64.b64decode(encoded, validate=True)
        except ValueError as error:
            raise ValueError(f"Invalid base64 image artifact: {uri}") from error
    content_type = str(response.get("ContentType") or "")
    if content_type and content_type != "image/jpeg":
        raise ValueError(f"Unsupported report image content type: {content_type}")
    return base64.b64encode(body).decode("ascii"), body


def _parse_spine(xml_text: str) -> dict[str, Any]:
    upper = xml_text.upper()
    if "<!DOCTYPE" in upper or "<!ENTITY" in upper:
        raise ValueError("Correlation XML must not contain DTD or entity declarations")
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as error:
        raise ValueError(f"Invalid correlation XML: {error}") from error
    summary = (root.findtext("summary") or "Correlation completed.").strip()
    source_image = (root.findtext("./metadata/source_image") or "").strip()
    if not source_image:
        raise ValueError("Correlation XML is missing metadata/source_image")
    specialists = [
        {"name": node.attrib.get("name", ""), "s3_uri": node.attrib.get("s3_uri", "")}
        for node in root.findall("./metadata/specialists_executed/specialist")
    ]
    elements: list[dict[str, Any]] = []

    def walk(node: ET.Element, depth: int = 0) -> None:
        for child in list(node):
            if child.tag == "element":
                text = (
                    child.findtext("text") or child.findtext("alt_text") or ""
                ).strip()
                elements.append(
                    {
                        "id": child.attrib.get("id", ""),
                        "tag": child.attrib.get("tag", "P"),
                        "order": child.attrib.get("order", ""),
                        "page": child.attrib.get("page", root.attrib.get("page", "")),
                        "text": text,
                        "depth": depth,
                    }
                )
                walk(child, depth + 1)
            elif child.tag == "sect":
                walk(child, depth + 1)
            else:
                # Wrapper nodes such as children and inline preserve structural
                # depth while still exposing their descendant elements.
                walk(child, depth)

    content_tree = root.find("content_tree")
    if content_tree is not None:
        walk(content_tree)
    return {
        "summary": summary,
        "source_image": source_image,
        "specialists": specialists,
        "elements": elements,
    }


def _page_audit(
    records: list[dict[str, Any]],
    image_uri: str,
    correlation_uri: str,
    specialists: list[dict[str, str]],
) -> list[dict[str, str]]:
    image_id = job_state.image_identifier(image_uri)
    result_uris = {item.get("s3_uri", "") for item in specialists}
    result_uris.add(correlation_uri)
    return [
        {
            "specialist": str(record.get("specialist") or ""),
            "status": str(record.get("status") or ""),
            "started_at": str(record.get("started_at") or ""),
            "completed_at": str(record.get("completed_at") or ""),
            "result_s3_key": str(record.get("result_s3_key") or ""),
            "error": str(record.get("error") or ""),
        }
        for record in records
        if record.get("image_identifier") == image_id
        or record.get("result_s3_key") in result_uris
    ]


def _manifest(
    report: dict[str, Any], html_key: str, manifest_key: str
) -> dict[str, Any]:
    manifest = {key: value for key, value in report.items() if key != "pages"}
    manifest["html_key"] = html_key
    manifest["manifest_key"] = manifest_key
    manifest["pages"] = [
        {key: value for key, value in page.items() if key not in {"image_data", "xml"}}
        for page in report["pages"]
    ]
    return manifest


def _owner_key(user_id: str) -> str:
    if user_id in {"local", "local-dev"}:
        return "local"
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:24]


def _safe_id(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "-", value).strip("-.")
    return cleaned[:80] or fallback


def _response(status_code: int, **body: Any) -> dict[str, Any]:
    return {"statusCode": status_code, "body": json.dumps(body)}
