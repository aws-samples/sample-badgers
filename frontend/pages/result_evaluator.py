#!/usr/bin/env python3
"""
Result Evaluator - Gradio app for evaluating analyzer outputs.

Lists session IDs from S3 output bucket, loads results with evaluation forms
based on manifest evaluation configurations.
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import boto3
import gradio as gr
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


def load_manifest_evaluation(analyzer_name: str) -> Optional[dict]:
    """Load evaluation config from analyzer manifest in S3."""
    if not S3_CONFIG_BUCKET:
        logger.warning("S3_CONFIG_BUCKET not set, cannot load manifest evaluation")
        return None

    s3 = boto3.client("s3", region_name=AWS_REGION)

    # Try base manifests first, then custom-analyzers
    keys_to_try = [
        f"manifests/{analyzer_name}.json",
        f"{CUSTOM_ANALYZERS_PREFIX}/manifests/{analyzer_name}.json",
    ]

    for key in keys_to_try:
        try:
            response = s3.get_object(Bucket=S3_CONFIG_BUCKET, Key=key)
            manifest: dict = json.loads(response["Body"].read().decode("utf-8"))
            eval_config: Optional[dict] = manifest.get("evaluation")
            logger.info(
                "Loaded evaluation config from s3://%s/%s: %s",
                S3_CONFIG_BUCKET,
                key,
                eval_config is not None,
            )
            return eval_config
        except s3.exceptions.NoSuchKey:
            continue
        except Exception as e:
            logger.error(
                "Error loading manifest from s3://%s/%s: %s", S3_CONFIG_BUCKET, key, e
            )
            continue

    logger.warning("No manifest found for analyzer: %s", analyzer_name)
    return None


def format_session_metadata(metadata: Optional[dict]) -> str:
    """Format session metadata as readable markdown."""
    if not metadata:
        return "*No session metadata available*"

    lines = []

    # Input file info
    input_file = metadata.get("input_file", {})
    file_name = input_file.get("name", "Unknown")
    lines.append(f"**File:** `{file_name}`")

    # Timestamp
    timestamp = metadata.get("timestamp_completed", "")
    if timestamp:
        # Format nicely if possible
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            lines.append(f"**Processed:** {formatted_time}")
        except (ValueError, AttributeError):
            lines.append(f"**Processed:** {timestamp}")

    # Stats summary
    stats = metadata.get("stats", {})
    total = stats.get("total_analyses_performed", 0)
    successful = stats.get("successful_analyses", 0)
    pages = stats.get("pages_with_content", 0)
    lines.append(f"**Analyses:** {successful}/{total} successful across {pages} pages")

    # Content summary
    content_summary = metadata.get("content_summary", {})
    if content_summary:
        lines.append("")
        lines.append("**Content Detected:**")
        for tool_name, info in content_summary.items():
            count = info.get("count", 0)
            page_list = info.get("pages", [])
            # Show unique pages and count if there are duplicates
            unique_pages = sorted(set(page_list))
            if len(page_list) > len(unique_pages):
                pages_str = f"pages {unique_pages} ({count} total)"
            else:
                pages_str = f"pages {unique_pages}"
            lines.append(f"- `{tool_name}`: {count} on {pages_str}")

    return "\n".join(lines)


def create_ui():
    """Create the Gradio interface."""

    if not S3_OUTPUT_BUCKET:
        with gr.Blocks(title="Result Evaluator") as demo_evaluator:
            gr.Markdown("# ❌ Configuration Error")
            gr.Markdown("S3_OUTPUT_BUCKET not set in .env file")
        return demo_evaluator

    # Lazy-init: don't create boto3 client at import time (credential chain is slow)
    s3_client_holder = [None]

    def get_s3_client():
        if s3_client_holder[0] is None:
            s3_client_holder[0] = S3Client(S3_OUTPUT_BUCKET, AWS_REGION)
        return s3_client_holder[0]

    with gr.Blocks(title="Result Evaluator") as demo_evaluator:
        # State
        current_session = gr.State("")
        current_results = gr.State([])
        current_index = gr.State(0)

        gr.Markdown("# 📊 Result Evaluator")

        gr.Markdown(f"Bucket: `{S3_OUTPUT_BUCKET}`")
        result_info = gr.Markdown("Select a session and click Load Session")

        gr.Markdown("---")

        with gr.Row(elem_classes=["content-panel"]):
            with gr.Column(scale=3):
                session_dropdown = gr.Dropdown(
                    choices=[],
                    label="Select Session",
                    info="Sessions from S3 output bucket (click Refresh to load)",
                )
            with gr.Column(scale=2):
                load_session_btn = gr.Button(
                    "📂 Load Session", variant="primary", size="sm"
                )
                refresh_btn = gr.Button(
                    "🔄 Refresh Session List",
                    size="sm",
                    variant="secondary",
                    elem_classes=["background-green-faded"],
                )

        # Session metadata (hidden until loaded)
        session_metadata_display = gr.Markdown(
            "",
            label="Session Info",
            visible=False,
        )

        gr.Markdown("---")

        gr.Markdown("## Move through files within this session.")
        with gr.Row():
            prev_btn = gr.Button(
                "◀ Previous", size="sm", elem_classes=["background-green-faded"]
            )
            progress_text = gr.Markdown("", elem_classes=["progress-counter"])
            next_btn = gr.Button(
                "Next ▶", size="sm", elem_classes=["background-green-faded"]
            )

        gr.Markdown("---")
        with gr.Row():
            # Left: XML Output
            with gr.Column(scale=1):
                gr.Markdown("### 📄 Result Output")
                result_display = gr.Textbox(
                    label="XML Content",
                    lines=25,
                    interactive=False,
                )

            # Right: Evaluation Form
            with gr.Column(scale=1):
                gr.Markdown("### ✅ Evaluation")
                analyzer_name_display = gr.Markdown("")

                # Core questions group
                with gr.Group():
                    gr.Markdown("#### Core Questions")

                    overall_accuracy = gr.Radio(
                        choices=[
                            "Failed",
                            "Major errors",
                            "Partial success",
                            "Minor issues",
                            "Perfect",
                        ],
                        label="Overall accuracy of the analysis",
                        type="index",
                    )
                    element_identification = gr.Radio(
                        choices=[
                            "Failed",
                            "Major errors",
                            "Partial success",
                            "Minor issues",
                            "Perfect",
                        ],
                        label="Were all visual elements correctly identified?",
                        type="index",
                    )
                    contextual_understanding = gr.Radio(
                        choices=[
                            "Failed",
                            "Major errors",
                            "Partial success",
                            "Minor issues",
                            "Perfect",
                        ],
                        label="Was the content understood within its surrounding context?",
                        type="index",
                    )
                    issues_noted = gr.Textbox(
                        label="What elements were missed or incorrectly represented?",
                        lines=3,
                    )

                gr.Markdown("---")

                # Tool-specific questions group (dynamic, hidden when no eval config)
                tool_specific_group = gr.Group(visible=False)
                with tool_specific_group:
                    gr.Markdown("#### Tool-Specific Questions")
                    tool_q1 = gr.Radio(
                        choices=[
                            "Failed",
                            "Major errors",
                            "Partial success",
                            "Minor issues",
                            "Perfect",
                        ],
                        label="Tool-specific question 1",
                        type="index",
                    )
                    tool_q2 = gr.Radio(
                        choices=[
                            "Failed",
                            "Major errors",
                            "Partial success",
                            "Minor issues",
                            "Perfect",
                        ],
                        label="Tool-specific question 2",
                        type="index",
                    )
                    tool_q3 = gr.Radio(
                        choices=[
                            "Failed",
                            "Major errors",
                            "Partial success",
                            "Minor issues",
                            "Perfect",
                        ],
                        label="Tool-specific question 3",
                        type="index",
                    )

                save_btn = gr.Button("💾 Save Evaluation", variant="primary")
                save_status = gr.Markdown("")

        # Event handlers
        def load_session(session_id):
            """Load results for selected session."""
            if not session_id:
                return (
                    "",
                    [],
                    0,
                    "Select a session",
                    "*Session metadata will appear here after loading*",
                    "",
                    "",
                    None,
                    None,
                    None,
                    "",
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    "",
                    gr.update(visible=False),
                )

            results = get_s3_client().list_results(session_id)
            # Fetch session metadata
            metadata = get_s3_client().get_session_metadata(session_id)
            metadata_display = format_session_metadata(metadata)

            if not results:
                return (
                    session_id,
                    [],
                    0,
                    "No results found",
                    metadata_display,
                    "",
                    "",
                    None,
                    None,
                    None,
                    "",
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    "",
                    gr.update(visible=False),
                )

            return load_result_at_index(session_id, results, 0, metadata_display)

        def load_result_at_index(session_id, results, index, metadata_display=""):
            """Load a specific result by index."""
            if not results or index < 0 or index >= len(results):
                return (
                    session_id,
                    results,
                    index,
                    "No results",
                    metadata_display,
                    "",
                    "",
                    None,
                    None,
                    None,
                    "",
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    "",
                    gr.update(visible=False),
                )

            result = results[index]
            content = get_s3_client().get_result_content(result["key"])
            analyzer = result["analyzer"]

            # Load evaluation config
            eval_config = load_manifest_evaluation(analyzer)
            logger.info(
                "Analyzer: %s, eval_config found: %s", analyzer, eval_config is not None
            )

            # Load existing evaluation
            existing_eval = get_s3_client().get_evaluation(
                session_id, result["filename"]
            )

            # Pre-fill from existing evaluation
            overall_val = None
            element_val = None
            context_val = None
            issues_val = ""
            tool_q1_val = None
            tool_q2_val = None
            tool_q3_val = None

            if existing_eval:
                responses = existing_eval.get("responses", {})
                overall_val = responses.get("overall_accuracy")
                element_val = responses.get("element_identification")
                context_val = responses.get("contextual_understanding")
                issues_val = responses.get("issues_noted", "")
                tool_q1_val = responses.get("tool_q1")
                tool_q2_val = responses.get("tool_q2")
                tool_q3_val = responses.get("tool_q3")

            # Prepare tool-specific questions — hide section if no eval config
            has_tool_questions = False
            tool_q1_update = gr.update(label="(Not applicable)", value=None)
            tool_q2_update = gr.update(label="(Not applicable)", value=None)
            tool_q3_update = gr.update(label="(Not applicable)", value=None)

            if eval_config and "questions" in eval_config:
                tool_specific = eval_config["questions"].get("tool_specific", [])
                logger.info("Tool-specific questions found: %d", len(tool_specific))
                if tool_specific:
                    has_tool_questions = True
                if len(tool_specific) > 0:
                    tool_q1_update = gr.update(
                        label=tool_specific[0]["text"],
                        value=tool_q1_val,
                    )
                if len(tool_specific) > 1:
                    tool_q2_update = gr.update(
                        label=tool_specific[1]["text"],
                        value=tool_q2_val,
                    )
                if len(tool_specific) > 2:
                    tool_q3_update = gr.update(
                        label=tool_specific[2]["text"],
                        value=tool_q3_val,
                    )
            else:
                logger.warning("No eval_config or questions for analyzer: %s", analyzer)

            tool_group_update = gr.update(visible=has_tool_questions)

            progress = f"<center style='font-size:2.5em;font-weight:700'>{index + 1}/{len(results)}</center>"
            info = f"**File:** {result['filename']} **Analyzer:** {analyzer}"
            analyzer_display = f"**Analyzer:** `{analyzer}`"

            return (
                session_id,
                results,
                index,
                info,
                metadata_display,
                content,
                analyzer_display,
                overall_val,
                element_val,
                context_val,
                issues_val,
                tool_q1_update,
                tool_q2_update,
                tool_q3_update,
                progress,
                tool_group_update,
            )

        def _save_current_to_session(
            session_id, results, index, overall, element, context, issues, tq1, tq2, tq3
        ):
            """Save current evaluation into the session-level file."""
            if not session_id or not results or index >= len(results):
                return
            # Only save if at least one field has been filled in
            if overall is None and element is None and context is None:
                return
            result = results[index]
            responses = {
                "overall_accuracy": overall,
                "element_identification": element,
                "contextual_understanding": context,
                "issues_noted": issues,
                "tool_q1": tq1,
                "tool_q2": tq2,
                "tool_q3": tq3,
            }
            try:
                get_s3_client().save_session_evaluation(
                    session_id, result["filename"], result["analyzer"], responses
                )
            except Exception as e:
                logger.warning("Auto-save failed: %s", e)

        def go_previous(
            session_id, results, index, overall, element, context, issues, tq1, tq2, tq3
        ):
            """Auto-save current evaluation, then navigate to previous result."""
            _save_current_to_session(
                session_id,
                results,
                index,
                overall,
                element,
                context,
                issues,
                tq1,
                tq2,
                tq3,
            )
            new_index = max(0, index - 1)
            return load_result_at_index(session_id, results, new_index)

        def go_next(
            session_id, results, index, overall, element, context, issues, tq1, tq2, tq3
        ):
            """Auto-save current evaluation, then navigate to next result."""
            _save_current_to_session(
                session_id,
                results,
                index,
                overall,
                element,
                context,
                issues,
                tq1,
                tq2,
                tq3,
            )
            new_index = min(len(results) - 1, index + 1)
            return load_result_at_index(session_id, results, new_index)

        def save_evaluation(
            session_id, results, index, overall, element, context, issues, tq1, tq2, tq3
        ):
            """Save the current evaluation to the session file."""
            if not session_id or not results or index >= len(results):
                return "❌ No result selected"

            result = results[index]
            responses = {
                "overall_accuracy": overall,
                "element_identification": element,
                "contextual_understanding": context,
                "issues_noted": issues,
                "tool_q1": tq1,
                "tool_q2": tq2,
                "tool_q3": tq3,
            }

            try:
                uri = get_s3_client().save_session_evaluation(
                    session_id, result["filename"], result["analyzer"], responses
                )
                return f"✅ Saved to `{uri}`"
            except Exception as e:
                return f"❌ Error: {e}"

        def refresh_sessions():
            """Refresh session list."""
            return gr.update(choices=get_s3_client().list_sessions())

        # Wire events
        outputs = [
            current_session,
            current_results,
            current_index,
            result_info,
            session_metadata_display,
            result_display,
            analyzer_name_display,
            overall_accuracy,
            element_identification,
            contextual_understanding,
            issues_noted,
            tool_q1,
            tool_q2,
            tool_q3,
            progress_text,
            tool_specific_group,
        ]

        load_session_btn.click(
            load_session,
            inputs=[session_dropdown],
            outputs=outputs,
        )

        nav_inputs = [
            current_session,
            current_results,
            current_index,
            overall_accuracy,
            element_identification,
            contextual_understanding,
            issues_noted,
            tool_q1,
            tool_q2,
            tool_q3,
        ]

        prev_btn.click(
            go_previous,
            inputs=nav_inputs,
            outputs=outputs,
        )

        next_btn.click(
            go_next,
            inputs=nav_inputs,
            outputs=outputs,
        )

        save_btn.click(
            save_evaluation,
            inputs=[
                current_session,
                current_results,
                current_index,
                overall_accuracy,
                element_identification,
                contextual_understanding,
                issues_noted,
                tool_q1,
                tool_q2,
                tool_q3,
            ],
            outputs=[save_status],
        )

        refresh_btn.click(refresh_sessions, outputs=[session_dropdown])

        # Lazy-load sessions after UI renders (non-blocking)
        demo_evaluator.load(refresh_sessions, outputs=[session_dropdown])

    return demo_evaluator


# Module-level demo for import
demo = create_ui()

if __name__ == "__main__":
    logger.info("Starting Result Evaluator")
    logger.info("S3_OUTPUT_BUCKET: %s", S3_OUTPUT_BUCKET)
    logger.info("AWS_REGION: %s", AWS_REGION)

    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
    )
