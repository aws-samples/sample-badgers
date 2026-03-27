#!/usr/bin/env python3
"""
S3Client for Sessions
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import boto3
from dotenv import load_dotenv

# Load environment
env_file = Path(__file__).parent.parent / "config" / ".env"
if env_file.exists():
    load_dotenv(env_file)

# Setup logging
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "evaluator.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Config from environment
S3_OUTPUT_BUCKET = os.getenv("S3_OUTPUT_BUCKET", "")
S3_CONFIG_BUCKET = os.getenv("S3_CONFIG_BUCKET", "")
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
CUSTOM_ANALYZERS_PREFIX = "custom-analyzers"


class S3Client:
    """S3 client for browsing and loading results."""

    def __init__(self, bucket: str, region: str):
        self.bucket = bucket
        self.region = region
        self.s3 = boto3.client("s3", region_name=region)
        logger.info("S3Client initialized: bucket=%s, region=%s", bucket, region)

    def list_sessions(self) -> list[str]:
        """List all session folders in the bucket."""
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.bucket,
                Delimiter="/",
            )
            sessions = []
            for prefix in response.get("CommonPrefixes", []):
                session_id = prefix["Prefix"].rstrip("/")
                # Skip evaluations folder
                if session_id != "evaluations":
                    sessions.append(session_id)
            logger.info("Found %d sessions", len(sessions))
            return sorted(sessions, reverse=True)
        except Exception as e:
            logger.error("Error listing sessions: %s", e)
            return []

    def list_results(self, session_id: str) -> list[dict]:
        """List all result files in a session folder (including subfolders)."""
        try:
            # Use paginator to list all objects recursively
            paginator = self.s3.get_paginator("list_objects_v2")
            results = []

            for page in paginator.paginate(Bucket=self.bucket, Prefix=f"{session_id}/"):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    # Skip evaluation files and non-xml files
                    if key.endswith(".xml") and "/evaluations/" not in key:
                        filename = key.split("/")[-1]
                        # Extract analyzer from path: session_id/analyzer_name/filename.xml
                        parts = key.split("/")
                        if len(parts) >= 3:
                            analyzer_name = parts[
                                1
                            ]  # The subfolder is the analyzer name
                        else:
                            analyzer_name = self._extract_analyzer_name(filename)

                        results.append(
                            {
                                "key": key,
                                "filename": filename,
                                "analyzer": analyzer_name,
                                "size": obj["Size"],
                                "modified": obj["LastModified"].isoformat(),
                            }
                        )
            logger.info("Found %d results in session %s", len(results), session_id)
            return results
        except Exception as e:
            logger.error("Error listing results: %s", e)
            return []

    def get_result_content(self, key: str) -> str:
        """Get content of a result file."""
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
            content: str = response["Body"].read().decode("utf-8")
            return content
        except Exception as e:
            logger.error("Error getting result: %s", e)
            return f"Error loading result: {e}"

    def save_evaluation(
        self, session_id: str, result_filename: str, evaluation: dict
    ) -> str:
        """Save evaluation to S3 with human-readable timestamp."""
        try:
            # Human-readable timestamp: 2026-02-03-02-30-PM
            timestamp = datetime.utcnow().strftime("%Y-%m-%d-%I-%M-%p")
            eval_key = (
                f"{session_id}/evaluations/{result_filename}_eval_{timestamp}.json"
            )
            evaluation["evaluated_at"] = datetime.utcnow().isoformat()
            evaluation["evaluated_at_readable"] = timestamp

            self.s3.put_object(
                Bucket=self.bucket,
                Key=eval_key,
                Body=json.dumps(evaluation, indent=2).encode("utf-8"),
                ContentType="application/json",
            )
            logger.info("Saved evaluation to %s", eval_key)
            return f"s3://{self.bucket}/{eval_key}"
        except Exception as e:
            logger.warning("Error saving evaluation: %s", e)
            raise

    def save_session_evaluation(
        self, session_id: str, result_filename: str, analyzer: str, evaluation: dict
    ) -> str:
        """Save/append evaluation into a single session-level evaluation file."""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d-%I-%M-%p")
        eval_key = f"{session_id}/evaluations/session_evaluation.json"

        # Try to load existing session evaluation file
        existing: dict = {
            "evaluations": [],
            "last_updated": "",
            "last_updated_readable": "",
        }
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=eval_key)
            existing = json.loads(response["Body"].read().decode("utf-8"))
            if "evaluations" not in existing:
                existing["evaluations"] = []
        except Exception:
            pass  # File doesn't exist yet, start fresh

        # Build entry for this result
        entry = {
            "result_file": result_filename,
            "analyzer": analyzer,
            "responses": evaluation,
            "evaluated_at": datetime.utcnow().isoformat(),
            "evaluated_at_readable": timestamp,
        }

        # Update or append: replace if same result_file already exists
        found = False
        for i, ev in enumerate(existing["evaluations"]):
            if ev.get("result_file") == result_filename:
                existing["evaluations"][i] = entry
                found = True
                break
        if not found:
            existing["evaluations"].append(entry)

        existing["last_updated"] = datetime.utcnow().isoformat()
        existing["last_updated_readable"] = timestamp

        self.s3.put_object(
            Bucket=self.bucket,
            Key=eval_key,
            Body=json.dumps(existing, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        logger.info(
            "Saved session evaluation to %s (%d results)",
            eval_key,
            len(existing["evaluations"]),
        )
        return f"s3://{self.bucket}/{eval_key}"

    def get_evaluation(self, session_id: str, result_filename: str) -> Optional[dict]:
        """Get most recent evaluation if it exists. Checks session file first, then per-file evals."""
        try:
            # Check session-level evaluation file first
            session_key = f"{session_id}/evaluations/session_evaluation.json"
            try:
                response = self.s3.get_object(Bucket=self.bucket, Key=session_key)
                session_data = json.loads(response["Body"].read().decode("utf-8"))
                for ev in session_data.get("evaluations", []):
                    if ev.get("result_file") == result_filename:
                        matched_eval: dict = ev
                        return matched_eval
            except Exception:
                pass

            # Fallback: check per-file evaluations
            prefix = f"{session_id}/evaluations/{result_filename}_eval_"
            response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)

            eval_files = [obj["Key"] for obj in response.get("Contents", [])]
            if not eval_files:
                # Try legacy format without timestamp
                legacy_key = f"{session_id}/evaluations/{result_filename}_eval.json"
                try:
                    response = self.s3.get_object(Bucket=self.bucket, Key=legacy_key)
                    result: dict = json.loads(response["Body"].read().decode("utf-8"))
                    return result
                except self.s3.exceptions.NoSuchKey:
                    return None

            # Get the most recent (last alphabetically due to timestamp format)
            latest_key = sorted(eval_files)[-1]
            response = self.s3.get_object(Bucket=self.bucket, Key=latest_key)
            latest_eval: dict = json.loads(response["Body"].read().decode("utf-8"))
            return latest_eval
        except Exception as e:
            logger.error("Error getting evaluation: %s", e)
            return None

    def get_session_metadata(self, session_id: str) -> Optional[dict]:
        """Get session metadata if it exists."""
        try:
            metadata_key = f"{session_id}/session_metadata.json"
            response = self.s3.get_object(Bucket=self.bucket, Key=metadata_key)
            result: dict = json.loads(response["Body"].read().decode("utf-8"))
            return result
        except self.s3.exceptions.NoSuchKey:
            logger.info("No session metadata found for %s", session_id)
            return None
        except Exception as e:
            logger.error("Error getting session metadata: %s", e)
            return None

    def _extract_analyzer_name(self, filename: str) -> str:
        """Extract analyzer name from result filename."""
        # Format: {analyzer_name}_{image_identifier}_{timestamp}.xml
        match = re.match(r"^([a-z_]+_analyzer)", filename)
        if match:
            return match.group(1)
        # Fallback: take first part before underscore
        return filename.split("_")[0] + "_analyzer"
