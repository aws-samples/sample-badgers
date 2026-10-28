"""Type-aware, opt-in Vision analysis for safe XLSX visual-object evidence.

Native XLSX parsing remains authoritative. This module only analyzes the
pixels of supported embedded images and joins its result to the native drawing
anchor and covered-cell evidence already produced by ``workbook_inspector``.
It never renders or opens a workbook, evaluates a formula, or treats a Vision
response as a replacement for native cell data.
"""

from __future__ import annotations

import json
import os
import posixpath
import re
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any


class VisualAnalysisError(ValueError):
    """Raised when visual analysis cannot be safely prepared or validated."""


MAX_VISUAL_ANALYSIS_OBJECTS = 5
MAX_VISUAL_ANALYSIS_IMAGE_BYTES = 5 * 1024 * 1024
SUPPORTED_IMAGE_CONTENT_TYPES = frozenset(
    {"image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp"}
)
SUPPORTED_ANALYSIS_KINDS = frozenset({"image", "chart"})
_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_GRADES = ((90, "A"), (80, "B"), (70, "C"), (60, "D"), (0, "F"))


def grade_for_confidence(confidence_percent: int) -> str:
    """Map a 0–100 confidence score to the project's school-style grade."""
    for threshold, grade in _GRADES:
        if confidence_percent >= threshold:
            return grade
    return "F"


def prepare_visual_analysis(
    workbook_bytes: bytes,
    inspection: dict[str, Any],
    *,
    max_objects: int = MAX_VISUAL_ANALYSIS_OBJECTS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Prepare bounded model inputs from native visual-object inventory.

    The present implementation sends actual embedded images to Vision. Native
    charts are recognized as a separate type and retained as pending until a
    deterministic, faithful chart renderer is added; it would be misleading to
    ask a vision model to infer chart pixels from OOXML metadata alone.
    """
    if not 1 <= max_objects <= MAX_VISUAL_ANALYSIS_OBJECTS:
        raise VisualAnalysisError(
            f"max_objects must be between 1 and {MAX_VISUAL_ANALYSIS_OBJECTS}."
        )

    prepared: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    with zipfile.ZipFile(BytesIO(workbook_bytes)) as archive:
        for sheet in inspection.get("sheets", []):
            if not isinstance(sheet, dict):
                continue
            sheet_name = str(sheet.get("name", ""))
            for visual_object in sheet.get("visual_objects", []):
                if not isinstance(visual_object, dict):
                    continue
                kind = str(visual_object.get("kind", ""))
                object_id = str(visual_object.get("object_id", "unknown"))
                if kind not in SUPPORTED_ANALYSIS_KINDS:
                    continue
                if kind == "chart":
                    skipped.append(
                        {
                            "object_id": object_id,
                            "kind": kind,
                            "status": "pending_rendered_chart_input",
                            "reason": (
                                "A native chart was detected, but no rendered chart "
                                "image is available for Vision analysis."
                            ),
                        }
                    )
                    continue
                if len(prepared) >= max_objects:
                    skipped.append(
                        {
                            "object_id": object_id,
                            "kind": kind,
                            "status": "not_analyzed",
                            "reason": "visual_analysis_object_limit",
                        }
                    )
                    continue

                try:
                    image_bytes, content_type = _embedded_image(archive, visual_object)
                except VisualAnalysisError as error:
                    skipped.append(
                        {
                            "object_id": object_id,
                            "kind": kind,
                            "status": "not_analyzed",
                            "reason": str(error),
                        }
                    )
                    continue

                prepared.append(
                    {
                        "sheet_name": sheet_name,
                        "object": visual_object,
                        "image_bytes": image_bytes,
                        "content_type": content_type,
                        "runtime_context": _runtime_context(sheet_name, visual_object),
                    }
                )
    return prepared, skipped


def analyze_workbook_visuals(
    workbook_bytes: bytes,
    inspection: dict[str, Any],
    *,
    system_prompt: str,
    model_config: dict[str, Any],
    max_objects: int = MAX_VISUAL_ANALYSIS_OBJECTS,
    aws_region: str | None = None,
) -> dict[str, Any]:
    """Run configured Bedrock Vision enrichment for supported visual objects."""
    selection = _model_selection(model_config)
    prepared, skipped = prepare_visual_analysis(
        workbook_bytes, inspection, max_objects=max_objects
    )
    analyses: list[dict[str, Any]] = []
    for item in prepared:
        visual_object = item["object"]
        analyses.append(
            invoke_visual_analysis(
                system_prompt=system_prompt,
                model_config=model_config,
                sheet_name=item["sheet_name"],
                visual_object=visual_object,
                image_bytes=item["image_bytes"],
                content_type=item["content_type"],
                runtime_context=item["runtime_context"],
                aws_region=aws_region,
            )
        )

    return {
        "schema_version": "badgers-xlsx-visual-analysis/v1",
        "status": "complete",
        "model_id": selection["model_id"],
        "analyses": analyses,
        "skipped_objects": skipped,
        "processing_guarantees": {
            "workbook_execution": "not_performed",
            "formula_calculation": "not_performed",
            "native_cells_authoritative": True,
            "vision_scope": "embedded_raster_visual_evidence_only",
        },
    }


def load_local_visual_configuration(
    manifest_path: Path, prompt_directory: Path
) -> tuple[str, dict[str, Any]]:
    """Load the checked-in prompt package for the explicit local sample runner."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        specialist = manifest["specialist"]
        prompt_files = specialist["prompt_files"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise VisualAnalysisError(
            f"Unable to load local XLSX Vision configuration from {manifest_path}."
        ) from error

    prompt_parts: list[str] = []
    for prompt_file in prompt_files:
        path = prompt_directory / str(prompt_file)
        try:
            prompt_parts.append(path.read_text(encoding="utf-8").strip())
        except OSError as error:
            raise VisualAnalysisError(f"Vision prompt file was not found: {path}") from error
    return "\n\n".join(part for part in prompt_parts if part), dict(specialist)


def load_deployed_visual_configuration(
    config_bucket: str, specialist_name: str = "xlsx_processor"
) -> tuple[str, dict[str, Any]]:
    """Load a release-pinned XLSX Vision prompt package from the config bucket."""
    if not config_bucket:
        raise VisualAnalysisError("CONFIG_BUCKET is required for Vision analysis.")
    try:
        from foundation.s3_config_loader import load_manifest_from_s3, load_prompt_from_s3

        manifest = load_manifest_from_s3(config_bucket, specialist_name)
        specialist = manifest["specialist"]
        prompt_files = specialist["prompt_files"]
        prompt_parts = [
            load_prompt_from_s3(config_bucket, specialist_name, str(prompt_file))
            for prompt_file in prompt_files
        ]
    except Exception as error:
        raise VisualAnalysisError(
            "Unable to load the release-pinned XLSX Vision configuration."
        ) from error
    return "\n\n".join(part.strip() for part in prompt_parts if part.strip()), dict(specialist)


def invoke_visual_analysis(
    *,
    system_prompt: str,
    model_config: dict[str, Any],
    sheet_name: str,
    visual_object: dict[str, Any],
    image_bytes: bytes,
    content_type: str,
    runtime_context: str,
    aws_region: str | None = None,
) -> dict[str, Any]:
    """Invoke the configured Bedrock model and validate its joined evidence."""
    if visual_object.get("kind") != "image":
        raise VisualAnalysisError(
            f"Vision input for kind {visual_object.get('kind')!r} is not implemented."
        )
    if content_type not in SUPPORTED_IMAGE_CONTENT_TYPES:
        raise VisualAnalysisError(f"Unsupported Bedrock Vision image type: {content_type}.")

    selection = _model_selection(model_config)
    try:
        import boto3

        response = boto3.Session().client(
            "bedrock-runtime", region_name=aws_region or "us-west-2"
        ).converse(
            modelId=os.environ.get("XLSX_VISION_MODEL") or selection["model_id"],
            system=[{"text": system_prompt}],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"text": runtime_context},
                        {
                            "image": {
                                "format": content_type.split("/", 1)[1].replace("jpg", "jpeg"),
                                "source": {"bytes": image_bytes},
                            }
                        },
                    ],
                }
            ],
            inferenceConfig={
                "maxTokens": selection["max_tokens"],
                "temperature": 0,
            },
        )
    except Exception as error:
        raise VisualAnalysisError("Bedrock Vision invocation failed.") from error

    text = "\n".join(
        str(block.get("text", ""))
        for block in response.get("output", {}).get("message", {}).get("content", [])
        if isinstance(block, dict) and "text" in block
    )
    return parse_visual_response(
        text,
        visual_object=visual_object,
        expected_object_id=str(visual_object.get("object_id", "")),
        expected_kind=str(visual_object.get("kind", "")),
        sheet_name=sheet_name,
    )


def parse_visual_response(
    value: str,
    *,
    visual_object: dict[str, Any],
    expected_object_id: str,
    expected_kind: str,
    sheet_name: str,
) -> dict[str, Any]:
    """Validate the intentionally small, non-authoritative Vision response."""
    try:
        parsed = json.loads(_JSON_FENCE.sub("", value.strip()).strip())
    except json.JSONDecodeError as error:
        raise VisualAnalysisError("Vision response was not valid JSON.") from error
    if not isinstance(parsed, dict):
        raise VisualAnalysisError("Vision response must be a JSON object.")

    required = {
        "object_id",
        "object_kind",
        "classification",
        "visual_summary",
        "visible_text",
        "relationship_to_covered_cells",
        "cell_visibility",
        "semantic_replacement_suspected",
        "confidence_percent",
        "confidence_grade",
        "requires_human_review",
    }
    missing = sorted(required - set(parsed))
    if missing:
        raise VisualAnalysisError("Vision response is missing: " + ", ".join(missing))
    if parsed["object_id"] != expected_object_id or parsed["object_kind"] != expected_kind:
        raise VisualAnalysisError("Vision response does not match the requested visual object.")
    if parsed["classification"] not in {
        "data_overlay",
        "annotation",
        "decorative",
        "unknown",
    }:
        raise VisualAnalysisError("Vision response has an unsupported classification.")
    if parsed["cell_visibility"] not in {
        "obscures",
        "partially_obscures",
        "does_not_obscure",
        "unknown",
    }:
        raise VisualAnalysisError("Vision response has an unsupported cell_visibility.")
    if not isinstance(parsed["semantic_replacement_suspected"], bool):
        raise VisualAnalysisError(
            "Vision semantic_replacement_suspected must be boolean."
        )
    confidence = parsed["confidence_percent"]
    if isinstance(confidence, bool) or not isinstance(confidence, int) or not 0 <= confidence <= 100:
        raise VisualAnalysisError("Vision confidence_percent must be an integer from 0 to 100.")
    if parsed["confidence_grade"] != grade_for_confidence(confidence):
        raise VisualAnalysisError("Vision confidence_grade does not match confidence_percent.")
    if not isinstance(parsed["visible_text"], list) or not all(
        isinstance(item, str) and len(item) <= 500 for item in parsed["visible_text"]
    ):
        raise VisualAnalysisError("Vision visible_text must be a list of short strings.")
    if not isinstance(parsed["requires_human_review"], bool):
        raise VisualAnalysisError("Vision requires_human_review must be boolean.")
    for field in ("visual_summary", "relationship_to_covered_cells"):
        if not isinstance(parsed[field], str) or len(parsed[field]) > 2_000:
            raise VisualAnalysisError(f"Vision {field} must be a short string.")

    return {
        "object_id": expected_object_id,
        "object_kind": expected_kind,
        "sheet_name": sheet_name,
        "classification": parsed["classification"],
        "visual_summary": parsed["visual_summary"],
        "visible_text": parsed["visible_text"][:30],
        "relationship_to_covered_cells": parsed["relationship_to_covered_cells"],
        "spatial_coverage": spatial_coverage(visual_object),
        "visual_effect": {
            "cell_visibility": parsed["cell_visibility"],
            "semantic_replacement_suspected": parsed[
                "semantic_replacement_suspected"
            ],
            "confidence_percent": confidence,
            "requires_human_review": parsed["requires_human_review"],
        },
        "confidence_percent": confidence,
        "confidence_grade": parsed["confidence_grade"],
        "requires_human_review": parsed["requires_human_review"],
        "analysis_scope": "non_authoritative_visual_evidence",
    }


def spatial_coverage(visual_object: dict[str, Any]) -> dict[str, Any]:
    """Return deterministic drawing-anchor coverage without inferring meaning."""
    anchor = visual_object.get("anchor")
    if not isinstance(anchor, dict):
        return {
            "relation": "anchor_unknown",
            "confidence": "deterministic",
        }
    start = anchor.get("from")
    end = anchor.get("to")
    if not (
        isinstance(start, dict)
        and isinstance(end, dict)
        and isinstance(start.get("column"), int)
        and isinstance(start.get("row"), int)
        and isinstance(end.get("column"), int)
        and isinstance(end.get("row"), int)
        and start["column"] > 0
        and start["row"] > 0
        and end["column"] > 0
        and end["row"] > 0
    ):
        return {
            "relation": "anchor_unknown",
            "confidence": "deterministic",
        }
    start_cell = _cell_coordinate(start["column"], start["row"])
    end_cell = _cell_coordinate(end["column"], end["row"])
    covered_cells = visual_object.get("covered_cells", [])
    return {
        "relation": "drawn_above_native_cells",
        "from": {
            "column": _column_label(start["column"]),
            "row": start["row"],
            "cell": start_cell,
        },
        "to": {
            "column": _column_label(end["column"]),
            "row": end["row"],
            "cell": end_cell,
        },
        "grid_span": f"{start_cell}:{end_cell}",
        "coordinate_delta": {
            "columns": abs(end["column"] - start["column"]),
            "rows": abs(end["row"] - start["row"]),
        },
        "covered_native_cell_count": (
            len(covered_cells) if isinstance(covered_cells, list) else 0
        ),
        "confidence": "deterministic",
    }


def _embedded_image(
    archive: zipfile.ZipFile, visual_object: dict[str, Any]
) -> tuple[bytes, str]:
    media_path = str(visual_object.get("media_path", ""))
    content_type = str(visual_object.get("media_content_type", ""))
    normalized = posixpath.normpath(media_path)
    if (
        not media_path
        or normalized != media_path
        or not normalized.startswith("xl/media/")
        or normalized.startswith("/")
    ):
        raise VisualAnalysisError("image_media_path_is_unsafe_or_missing")
    if content_type not in SUPPORTED_IMAGE_CONTENT_TYPES:
        raise VisualAnalysisError(f"unsupported_visual_media_type:{content_type or 'unknown'}")
    try:
        image_bytes = archive.read(normalized)
    except KeyError as error:
        raise VisualAnalysisError("image_media_missing") from error
    if not image_bytes:
        raise VisualAnalysisError("image_media_empty")
    if len(image_bytes) > MAX_VISUAL_ANALYSIS_IMAGE_BYTES:
        raise VisualAnalysisError("image_media_exceeds_vision_size_limit")
    return image_bytes, content_type


def _model_selection(model_config: dict[str, Any]) -> dict[str, Any]:
    """Read the small fixed model contract without depending on shared runtime code."""
    selections = model_config.get("model_selections")
    if not isinstance(selections, dict):
        raise VisualAnalysisError("XLSX Vision model_selections are required.")
    primary = selections.get("primary")
    model_id = primary.get("model_id") if isinstance(primary, dict) else primary
    if not isinstance(model_id, str) or not model_id:
        raise VisualAnalysisError("XLSX Vision primary model_id is required.")
    fallback_list = selections.get("fallback_list", [])
    if fallback_list:
        raise VisualAnalysisError(
            "XLSX Vision fallback models are not supported by this isolated component."
        )
    max_tokens = model_config.get("expected_output_tokens", 1200)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= 4096:
        raise VisualAnalysisError(
            "XLSX Vision expected_output_tokens must be an integer from 1 to 4096."
        )
    return {"model_id": model_id, "max_tokens": max_tokens}


def _cell_coordinate(column: int, row: int) -> str:
    return f"{_column_label(column)}{row}"


def _column_label(column: int) -> str:
    value = column
    label = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        label = chr(65 + remainder) + label
    return label


def _runtime_context(sheet_name: str, visual_object: dict[str, Any]) -> str:
    """Build a bounded fact-only join context for a Vision request."""
    context = {
        "task": "Analyze one embedded workbook image as non-authoritative evidence.",
        "object_kind": visual_object.get("kind"),
        "object_id": visual_object.get("object_id"),
        "sheet_name": sheet_name,
        "drawing_name": visual_object.get("name", ""),
        "drawing_description": visual_object.get("description", ""),
        "anchor": visual_object.get("anchor", {}),
        "spatial_coverage": spatial_coverage(visual_object),
        "covered_native_cells": visual_object.get("covered_cells", []),
        "constraints": [
            "Do not calculate formulas or infer hidden cell values.",
            "Do not replace, correct, or invent native workbook data.",
            "Only describe pixels present in the provided embedded image.",
            "Use covered native cells only to explain possible spatial relationship.",
        ],
    }
    return json.dumps(context, ensure_ascii=False, sort_keys=True)
