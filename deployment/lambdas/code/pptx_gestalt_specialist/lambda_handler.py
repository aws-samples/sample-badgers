"""Interpret a deterministic PPTX slide scene graph without looking at pixels.

The specialist never replaces a native PPTX spine. It derives geometry facts
in Python, then asks a text model to express only evidence-backed relationship
enrichments that can be attached to that canonical spine.
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timezone
from itertools import combinations
from typing import Any
from xml.etree import ElementTree

import boto3

from foundation import job_state
from foundation.bedrock_client import BedrockClient
from foundation.model_selection import max_tokens_from, parse_model_selection
from foundation.s3_config_loader import load_manifest_from_s3

logger = logging.getLogger()
logger.setLevel(getattr(logging, os.environ.get("LOGGING_LEVEL", "INFO").upper()))

_MAX_SHAPES = 300
_MAX_TEXT_CHARS = 60_000
_ALIGN_TOLERANCE = 0.015  # Scene-graph coordinates are normalized 0..1.
_PROXIMITY_TOLERANCE = 0.035


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:  # noqa: ARG001
    job_id = ""
    subtask = ""
    try:
        body = json.loads(event["body"]) if "body" in event else event
        session_id = str(body.get("session_id") or "no_session")
        slide_number = int(body.get("slide_number") or 0)
        source_document_path = str(body.get("source_document_path") or "")
        scene_graph = _load_scene_graph(body)
        if slide_number < 1 or not source_document_path:
            raise ValueError("slide_number and source_document_path are required")
        _validate_scene_graph(scene_graph, slide_number)

        specialist_name = os.environ.get(
            "SPECIALIST_NAME", "pptx_gestalt_specialist"
        )
        job_id, doc_id = str(body.get("job_id") or ""), str(body.get("doc_id") or "")
        subtask = job_state.subtask_id(
            specialist_name, f"{source_document_path}#slide-{slide_number}"
        )
        job_state.mark_running(
            job_id, subtask, doc_id=doc_id, specialist=specialist_name,
            image_id=job_state.image_identifier(source_document_path),
            session_id=session_id,
        )

        evidence = _derive_evidence(scene_graph)
        config_bucket = os.environ["CONFIG_BUCKET"]
        manifest = load_manifest_from_s3(config_bucket, specialist_name)
        config = manifest.get("specialist", manifest)
        prompt = _load_prompt(config_bucket, config)
        response = _interpret(prompt, slide_number, scene_graph, evidence, config)
        result = _extract_text(response)
        _validate_model_result(result, slide_number, scene_graph)
        output_bucket = os.environ["OUTPUT_BUCKET"]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        evidence_uri = _save_evidence(
            output_bucket,
            session_id,
            source_document_path,
            slide_number,
            timestamp,
            scene_graph,
            evidence,
        )
        key = (
            f"{session_id}/pptx_gestalt/"
            f"pptx_gestalt_slide_{slide_number:03d}_{timestamp}.xml"
        )
        boto3.client("s3").put_object(
            Bucket=output_bucket,
            Key=key,
            Body=result.encode("utf-8"),
            ContentType="application/xml",
            Tagging=f"session_id={session_id}&type=pptx_gestalt&page={slide_number}",
        )
        uri = f"s3://{output_bucket}/{key}"
        job_state.mark_complete(job_id, subtask, uri)
        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "result": result,
                    "pptx_gestalt_uri": uri,
                    "deterministic_evidence_uri": evidence_uri,
                    "success": True,
                    "session_id": session_id,
                }
            ),
        }
    except Exception as error:
        logger.error("PPTX Gestalt analysis failed: %s", error, exc_info=True)
        job_state.mark_failed(job_id, subtask, str(error))
        return {
            "statusCode": 400,
            "body": json.dumps(
                {"error_type": "PPTX_GESTALT_FAILED", "message": str(error)}
            ),
        }


def _load_scene_graph(body: dict[str, Any]) -> dict[str, Any]:
    graph = body.get("slide_scene_graph")
    if isinstance(graph, str):
        graph = json.loads(graph)
    if isinstance(graph, dict):
        return graph
    uri = str(body.get("scene_graph_uri") or "")
    if not uri.startswith("s3://"):
        raise ValueError("Provide slide_scene_graph JSON or scene_graph_uri")
    bucket, key = uri.removeprefix("s3://").split("/", 1)
    return json.loads(
        boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
    )


def _validate_scene_graph(graph: dict[str, Any], slide_number: int) -> None:
    graph_slide = graph.get("slide_number")
    if graph_slide is not None and int(graph_slide) != slide_number:
        raise ValueError("slide_scene_graph.slide_number does not match slide_number")
    shapes = graph.get("shapes")
    if not isinstance(shapes, list) or not shapes:
        raise ValueError("slide_scene_graph.shapes must be a non-empty array")
    if len(shapes) > _MAX_SHAPES:
        raise ValueError(f"slide_scene_graph supports at most {_MAX_SHAPES} shapes")
    if sum(len(str(shape.get("text") or "")) for shape in shapes) > _MAX_TEXT_CHARS:
        raise ValueError("slide_scene_graph text exceeds the 60,000-character limit")
    ids = set()
    for shape in shapes:
        if not isinstance(shape, dict) or not isinstance(shape.get("id"), str) or not shape["id"]:
            raise ValueError("Every scene-graph shape needs a non-empty id")
        if shape["id"] in ids:
            raise ValueError(f"Duplicate scene-graph shape id: {shape['id']}")
        ids.add(shape["id"])
        for key in ("x", "y", "width", "height"):
            value = shape.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Shape {shape['id']} is missing numeric {key}")
            if not math.isfinite(value):
                raise ValueError(f"Shape {shape['id']} has non-finite {key}")
        if (
            shape["x"] < 0
            or shape["y"] < 0
            or shape["width"] <= 0
            or shape["height"] <= 0
            or shape["x"] + shape["width"] > 1.0
            or shape["y"] + shape["height"] > 1.0
        ):
            raise ValueError(
                f"Shape {shape['id']} bounds must be normalized within the slide (0..1)"
            )


def _derive_evidence(graph: dict[str, Any]) -> dict[str, Any]:
    """Calculate, rather than infer, relationship candidates from coordinates."""
    shapes = graph["shapes"]
    shape_by_id = {str(shape["id"]): shape for shape in shapes}
    alignment, proximity, containment, connections = [], [], [], []
    relationship_shapes = [
        shape for shape in shapes if str(shape.get("type") or "").lower() != "connector"
    ]
    for left, right in combinations(relationship_shapes, 2):
        left_id, right_id = str(left["id"]), str(right["id"])
        if abs(left["x"] - right["x"]) <= _ALIGN_TOLERANCE:
            alignment.append({"kind": "left_aligned", "shapes": [left_id, right_id]})
        if abs(left["y"] - right["y"]) <= _ALIGN_TOLERANCE:
            alignment.append({"kind": "top_aligned", "shapes": [left_id, right_id]})
        if _edge_distance(left, right) <= _PROXIMITY_TOLERANCE:
            proximity.append({"shapes": [left_id, right_id]})
        if _contains(left, right):
            containment.append({"container": left_id, "member": right_id})
        elif _contains(right, left):
            containment.append({"container": right_id, "member": left_id})
    for shape in shapes:
        if str(shape.get("type") or "").lower() != "connector":
            continue
        source, target = str(shape.get("from_shape") or ""), str(shape.get("to_shape") or "")
        if source in shape_by_id and target in shape_by_id:
            connections.append(
                {
                    "connector": str(shape["id"]),
                    "from_shape": source,
                    "to_shape": target,
                    "direction": str(shape.get("direction") or "unspecified"),
                }
            )
    return {
        "alignment_pairs": alignment,
        "proximity_pairs": proximity,
        "containment_pairs": containment,
        "connector_edges": connections,
        "shape_count": len(shapes),
    }


def _contains(container: dict[str, Any], member: dict[str, Any]) -> bool:
    if container["id"] == member["id"]:
        return False
    return (
        container["x"] <= member["x"]
        and container["y"] <= member["y"]
        and container["x"] + container["width"] >= member["x"] + member["width"]
        and container["y"] + container["height"] >= member["y"] + member["height"]
        and (container["width"] * container["height"])
        > (member["width"] * member["height"]) * 1.15
    )


def _edge_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    horizontal = max(
        left["x"] - (right["x"] + right["width"]),
        right["x"] - (left["x"] + left["width"]),
        0,
    )
    vertical = max(
        left["y"] - (right["y"] + right["height"]),
        right["y"] - (left["y"] + left["height"]),
        0,
    )
    return max(horizontal, vertical)


def _load_prompt(bucket: str, config: dict[str, Any]) -> str:
    files = config.get("prompt_files") or []
    if not files:
        raise ValueError("PPTX Gestalt manifest has no prompt files")
    s3 = boto3.client("s3")
    return "\n".join(
        s3.get_object(
            Bucket=bucket,
            Key=f"prompts/pptx_gestalt_specialist/{filename}",
        )["Body"].read().decode("utf-8")
        for filename in files
    )


def _save_evidence(
    output_bucket: str,
    session_id: str,
    source_document_path: str,
    slide_number: int,
    timestamp: str,
    graph: dict[str, Any],
    evidence: dict[str, Any],
) -> str:
    """Persist the calculated basis so every model relationship remains auditable."""
    key = (
        f"{session_id}/pptx_gestalt/"
        f"pptx_gestalt_evidence_slide_{slide_number:03d}_{timestamp}.json"
    )
    artifact = {
        "schema_version": "1.0",
        "analysis_mode": "native_coordinate_gestalt",
        "source_document_path": source_document_path,
        "slide_number": slide_number,
        "scene_graph": graph,
        "deterministic_evidence": evidence,
    }
    boto3.client("s3").put_object(
        Bucket=output_bucket,
        Key=key,
        Body=json.dumps(artifact, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json",
        Tagging=f"session_id={session_id}&type=pptx_gestalt_evidence&page={slide_number}",
    )
    return f"s3://{output_bucket}/{key}"


def _interpret(
    prompt: str,
    slide_number: int,
    graph: dict[str, Any],
    evidence: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    selection = parse_model_selection(config)
    client = BedrockClient(
        throttle_delay=float(os.environ.get("THROTTLE_DELAY", "1.0")),
        aws_region=os.environ.get("AWS_REGION", "us-west-2"),
    )
    message = (
        f"Interpret native PowerPoint slide {slide_number}. Do not claim any "
        "relationship that lacks IDs in deterministic evidence. Set the XML root "
        f"slide attribute to {slide_number}.\n\nSCENE_GRAPH:\n"
        f"{json.dumps(graph, ensure_ascii=False)}\n\nDETERMINISTIC_EVIDENCE:\n"
        f"{json.dumps(evidence, ensure_ascii=False)}"
    )
    payload = client.create_anthropic_payload(
        system_prompt=prompt,
        messages=[{"role": "user", "content": message}],
        max_tokens=max_tokens_from(config, default=4000),
        temperature=0.1,
    )
    return client.invoke_model(
        model_id=selection["model_id"],
        payload=payload,
        fallback_list=selection["fallback_list"],
        max_retries=config.get("max_retries", 1),
        extended_thinking=selection["extended_thinking"],
        budget_tokens=selection["budget_tokens"],
        adaptive_thinking=selection["adaptive_thinking"],
        adaptive_effort=selection["effort"],
    )


def _extract_text(response: dict[str, Any]) -> str:
    for block in response.get("content", []):
        if block.get("type") == "text" and str(block.get("text") or "").strip():
            return str(block["text"]).strip()
    raise ValueError("Model returned no PPTX Gestalt result")


def _validate_model_result(
    result: str, slide_number: int, graph: dict[str, Any]
) -> None:
    # Keep the model bounded to an additive enrichment artifact, never a replacement
    # for the native BADGERS content tree.
    try:
        root = ElementTree.fromstring(result)
    except ElementTree.ParseError as error:
        raise ValueError("Model result is not valid XML") from error
    if root.tag != "pptx_gestalt_analysis":
        raise ValueError("Model result root must be pptx_gestalt_analysis")
    if root.attrib.get("slide") != str(slide_number):
        raise ValueError("Model result slide number does not match the request")
    if root.attrib.get("analysis_mode") != "native_coordinate_gestalt":
        raise ValueError("Model result must declare native_coordinate_gestalt mode")
    if root.find(".//content_tree") is not None:
        raise ValueError("Model result must not contain a replacement content_tree")
    if root.find("./relationships") is None:
        raise ValueError("Model result must include a relationships element")
    if root.find("./uncertainties") is None:
        raise ValueError("Model result must include an uncertainties element")
    visual_recommendation = root.find("./recommended_visual_analysis")
    if visual_recommendation is None:
        raise ValueError("Model result must state whether visual analysis is needed")
    if visual_recommendation.attrib.get("required") not in {"true", "false"}:
        raise ValueError(
            "recommended_visual_analysis.required must be true or false"
        )

    shape_ids = {str(shape["id"]) for shape in graph["shapes"]}
    for relationship in root.findall("./relationships/relationship"):
        required = ("id", "type", "confidence", "source_shape", "target_shape")
        if any(not relationship.attrib.get(attribute) for attribute in required):
            raise ValueError(
                "Each relationship must have id, type, confidence, source_shape, "
                "and target_shape attributes"
            )
        source = relationship.attrib.get("source_shape")
        target = relationship.attrib.get("target_shape")
        if source not in shape_ids or target not in shape_ids:
            raise ValueError(
                "Each relationship must reference known source_shape and target_shape IDs"
            )
        if source == target:
            raise ValueError("A relationship must reference two distinct shapes")
        evidence = relationship.findall("./evidence")
        if not evidence:
            raise ValueError("Each relationship must include deterministic evidence")
        evidence_text = " ".join(
            " ".join(evidence_item.itertext())
            + " "
            + " ".join(evidence_item.attrib.values())
            for evidence_item in evidence
        )
        if source not in evidence_text or target not in evidence_text:
            raise ValueError(
                "Relationship evidence must cite both source_shape and target_shape IDs"
            )
