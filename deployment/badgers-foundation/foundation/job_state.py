"""Shared DynamoDB job state helper for BADGERS.

Every writer — the orchestrator runtime, the built-in specialist Lambdas, and
the generated custom specialist Lambdas — goes through this module so the record
shape stays consistent in one place.

Hierarchy (see deployment/stacks/dynamodb_stack.py for the full rationale):

    doc_id  ->  job_id  ->  subtask_id

Table schema:
  PK  job_id      (String)
  SK  subtask_id  (String)  'orchestrator' for the job-level row, otherwise
                            '{specialist}#{image_identifier}'

The subtask sort key is deterministic rather than a UUID. That gives uniqueness
across the page fan-out (BADGERS invokes the same specialist once per page)
while making retries idempotent: re-running the same specialist against the same
page upserts the same row and increments retry_count rather than creating a
duplicate.

Every write is a no-op when JOBS_TABLE_NAME is unset, so specialists keep
working unchanged in deployments (or local runs) without job tracking.
"""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

ORCHESTRATOR_SUBTASK = "orchestrator"
_TTL_DAYS = 30

_dynamodb = None


def _table_name() -> str:
    # Read at call time, not import time: the Lambda environment may be
    # populated after this module is first imported.
    return os.environ.get("JOBS_TABLE_NAME", "")


def enabled() -> bool:
    """True when job tracking is configured for this deployment."""
    return bool(_table_name())


def _table():
    global _dynamodb
    if _dynamodb is None:
        import boto3

        region = (
            os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or "us-west-2"
        )
        _dynamodb = boto3.resource("dynamodb", region_name=region)
    return _dynamodb.Table(_table_name())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ttl() -> int:
    return int(time.time()) + (_TTL_DAYS * 86400)


def image_identifier(image_path: str | None) -> str:
    """Derive a stable, key-safe page/image identifier from a source path.

    Mirrors foundation.s3_result_saver._extract_image_identifier so a subtask id
    lines up with the S3 key the result is written to.
    """
    if not image_path:
        return "unknown"
    stem = image_path.rstrip("/").split("/")[-1]
    if "." in stem:
        stem = stem.rsplit(".", 1)[0]
    # '#' is the subtask_id separator; strip anything awkward for a sort key.
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", stem)
    return stem or "unknown"


def subtask_id(specialist: str, image_path: str | None = None) -> str:
    """Build the deterministic subtask sort key for a specialist invocation."""
    return f"{specialist}#{image_identifier(image_path)}"


# ── Write helpers ───────────────────────────────────────────────────


def mark_running(
    job_id: str,
    subtask: str,
    *,
    doc_id: str = "",
    specialist: str = "",
    image_id: str = "",
    session_id: str = "",
) -> None:
    """Upsert a record and transition it to RUNNING.

    update_item is an upsert, so no PENDING record needs to exist first.
    """
    if not (enabled() and job_id and subtask):
        return

    sets = ["#s = :s", "started_at = :t", "#ttl = :ttl"]
    names: dict[str, str] = {"#s": "status", "#ttl": "ttl"}
    values: dict[str, Any] = {":s": "RUNNING", ":t": _now_iso(), ":ttl": _ttl()}

    for attr, val in (
        ("doc_id", doc_id),
        ("specialist", specialist),
        ("image_identifier", image_id),
        ("session_id", session_id),
    ):
        if val:
            sets.append(f"{attr} = :{attr}")
            values[f":{attr}"] = val

    try:
        _table().update_item(
            Key={"job_id": job_id, "subtask_id": subtask},
            UpdateExpression="SET " + ", ".join(sets),
            ExpressionAttributeNames=names,
            ExpressionAttributeValues=values,
        )
    except Exception as e:  # never fail the analysis because tracking failed
        logger.warning("job_state.mark_running failed (job=%s): %s", job_id, e)


def mark_complete(job_id: str, subtask: str, result_s3_key: str = "") -> None:
    """Transition a record to COMPLETE and record the S3 output key."""
    if not (enabled() and job_id and subtask):
        return
    try:
        _table().update_item(
            Key={"job_id": job_id, "subtask_id": subtask},
            UpdateExpression=(
                "SET #s = :s, completed_at = :t, result_s3_key = :k, #ttl = :ttl"
            ),
            ExpressionAttributeNames={"#s": "status", "#ttl": "ttl"},
            ExpressionAttributeValues={
                ":s": "COMPLETE",
                ":t": _now_iso(),
                ":k": result_s3_key,
                ":ttl": _ttl(),
            },
        )
    except Exception as e:
        logger.warning("job_state.mark_complete failed (job=%s): %s", job_id, e)


def mark_failed(job_id: str, subtask: str, error: str) -> None:
    """Transition a record to FAILED."""
    if not (enabled() and job_id and subtask):
        return
    try:
        _table().update_item(
            Key={"job_id": job_id, "subtask_id": subtask},
            UpdateExpression="SET #s = :s, completed_at = :t, #e = :e, #ttl = :ttl",
            ExpressionAttributeNames={"#s": "status", "#e": "error", "#ttl": "ttl"},
            ExpressionAttributeValues={
                ":s": "FAILED",
                ":t": _now_iso(),
                ":e": str(error)[:1024],
                ":ttl": _ttl(),
            },
        )
    except Exception as e:
        logger.warning("job_state.mark_failed failed (job=%s): %s", job_id, e)


def create_job(
    job_id: str,
    doc_id: str = "",
    session_id: str = "",
    reason: str = "",
) -> None:
    """Create the job-level ('orchestrator') row if it does not already exist."""
    if not (enabled() and job_id):
        return

    item: dict[str, Any] = {
        "job_id": job_id,
        "subtask_id": ORCHESTRATOR_SUBTASK,
        "specialist": ORCHESTRATOR_SUBTASK,
        "status": "PENDING",
        "retry_count": 0,
        "started_at": _now_iso(),
        "ttl": _ttl(),
    }
    for attr, val in (
        ("doc_id", doc_id),
        ("session_id", session_id),
        ("reason", reason),
    ):
        if val:
            item[attr] = val

    try:
        _table().put_item(
            Item=item,
            ConditionExpression="attribute_not_exists(job_id)",
        )
    except Exception as e:
        # ConditionalCheckFailedException just means the job already exists.
        if type(e).__name__ != "ConditionalCheckFailedException" and (
            "ConditionalCheckFailed" not in str(e)
        ):
            logger.warning("job_state.create_job failed (job=%s): %s", job_id, e)


def increment_retry(job_id: str, subtask: str) -> int:
    """Atomically increment retry_count. Returns the new value (0 if disabled)."""
    if not (enabled() and job_id and subtask):
        return 0
    try:
        resp = _table().update_item(
            Key={"job_id": job_id, "subtask_id": subtask},
            UpdateExpression="ADD retry_count :one",
            ExpressionAttributeValues={":one": 1},
            ReturnValues="UPDATED_NEW",
        )
        return int(resp["Attributes"]["retry_count"])
    except Exception as e:
        logger.warning("job_state.increment_retry failed (job=%s): %s", job_id, e)
        return 0


# ── Read helpers ────────────────────────────────────────────────────


def get_record(job_id: str, subtask: str) -> dict[str, Any]:
    """Fetch a single record. Returns {} when missing or disabled."""
    if not (enabled() and job_id and subtask):
        return {}
    try:
        resp = _table().get_item(Key={"job_id": job_id, "subtask_id": subtask})
        return resp.get("Item", {}) or {}
    except Exception as e:
        logger.warning("job_state.get_record failed (job=%s): %s", job_id, e)
        return {}


def get_job_records(job_id: str) -> list[dict[str, Any]]:
    """Fetch every record for a job_id (the job row plus all subtasks)."""
    if not (enabled() and job_id):
        return []
    try:
        resp = _table().query(
            KeyConditionExpression="job_id = :jid",
            ExpressionAttributeValues={":jid": job_id},
        )
        return resp.get("Items", []) or []
    except Exception as e:
        logger.warning("job_state.get_job_records failed (job=%s): %s", job_id, e)
        return []
