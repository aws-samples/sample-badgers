"""Interpret native DOCX structure and optional layout facts without using pixels.

The canonical BADGERS spine comes from the DOCX parser. This specialist derives
auditable OOXML and layout evidence, then produces additive relationship
candidates. It never extracts a replacement spine or rasterizes a page.
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

_MAX_NODES = 500
_MAX_TEXT_CHARS = 100_000
_ALIGN_TOLERANCE = 0.015
_PROXIMITY_TOLERANCE = 0.035


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:  # noqa: ARG001
    job_id = ""
    subtask = ""
    try:
        body = json.loads(event["body"]) if "body" in event else event
        session_id = str(body.get("session_id") or "no_session")
        source_document_path = str(body.get("source_document_path") or "")
        scope_id = str(body.get("scope_id") or "document")
        page_number = body.get("page_number")
        if page_number is not None:
            page_number = int(page_number)
            if page_number < 1:
                raise ValueError("page_number must be a positive integer when supplied")
        graph = _load_graph(body)
        if not source_document_path:
            raise ValueError("source_document_path is required")
        _validate_graph(graph, page_number)

        specialist_name = os.environ.get(
            "SPECIALIST_NAME", "docx_structure_specialist"
        )
        job_id, doc_id = str(body.get("job_id") or ""), str(body.get("doc_id") or "")
        subtask = job_state.subtask_id(
            specialist_name, f"{source_document_path}#{scope_id}"
        )
        job_state.mark_running(
            job_id,
            subtask,
            doc_id=doc_id,
            specialist=specialist_name,
            image_id=job_state.image_identifier(source_document_path),
            session_id=session_id,
        )

        evidence = _derive_evidence(graph)
        config_bucket = os.environ["CONFIG_BUCKET"]
        manifest = load_manifest_from_s3(config_bucket, specialist_name)
        config = manifest.get("specialist", manifest)
        response = _interpret(
            _load_prompt(config_bucket, config),
            scope_id,
            page_number,
            graph,
            evidence,
            config,
        )
        result = _extract_text(response)
        _validate_model_result(result, scope_id, page_number, graph)

        output_bucket = os.environ["OUTPUT_BUCKET"]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        evidence_uri = _save_evidence(
            output_bucket,
            session_id,
            source_document_path,
            scope_id,
            page_number,
            timestamp,
            graph,
            evidence,
        )
        key = f"{session_id}/docx_structure/docx_structure_{scope_id}_{timestamp}.xml"
        boto3.client("s3").put_object(
            Bucket=output_bucket,
            Key=key,
            Body=result.encode("utf-8"),
            ContentType="application/xml",
            Tagging=f"session_id={session_id}&type=docx_structure",
        )
        uri = f"s3://{output_bucket}/{key}"
        job_state.mark_complete(job_id, subtask, uri)
        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "result": result,
                    "docx_structure_uri": uri,
                    "deterministic_evidence_uri": evidence_uri,
                    "success": True,
                    "session_id": session_id,
                }
            ),
        }
    except Exception as error:
        logger.error("DOCX structure analysis failed: %s", error, exc_info=True)
        job_state.mark_failed(job_id, subtask, str(error))
        return {
            "statusCode": 400,
            "body": json.dumps(
                {"error_type": "DOCX_STRUCTURE_FAILED", "message": str(error)}
            ),
        }


def _load_graph(body: dict[str, Any]) -> dict[str, Any]:
    graph = body.get("document_structure_graph")
    if isinstance(graph, str):
        graph = json.loads(graph)
    if isinstance(graph, dict):
        return graph
    uri = str(body.get("structure_graph_uri") or "")
    if not uri.startswith("s3://"):
        raise ValueError("Provide document_structure_graph JSON or structure_graph_uri")
    bucket, key = uri.removeprefix("s3://").split("/", 1)
    return json.loads(
        boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
    )


def _validate_graph(graph: dict[str, Any], page_number: int | None) -> None:
    nodes = graph.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("document_structure_graph.nodes must be a non-empty array")
    if len(nodes) > _MAX_NODES:
        raise ValueError(f"document_structure_graph supports at most {_MAX_NODES} nodes")
    if sum(len(str(node.get("text") or "")) for node in nodes) > _MAX_TEXT_CHARS:
        raise ValueError("document_structure_graph text exceeds the 100,000-character limit")
    graph_page = graph.get("page_number")
    if graph_page is not None and (
        isinstance(graph_page, bool)
        or not isinstance(graph_page, int)
        or graph_page < 1
    ):
        raise ValueError("document_structure_graph.page_number must be positive")
    if page_number is not None and graph_page is not None and graph_page != page_number:
        raise ValueError(
            "document_structure_graph.page_number does not match requested page_number"
        )

    ids = set()
    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError("Every document-structure node must be an object")
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("Every document-structure node needs a non-empty string id")
        if node_id in ids:
            raise ValueError(f"Duplicate document-structure node id: {node_id}")
        ids.add(node_id)
        if not isinstance(node.get("type"), str) or not node["type"]:
            raise ValueError(f"Node {node_id} needs a non-empty string type")
        order = node.get("order")
        if isinstance(order, bool) or not isinstance(order, (int, float)):
            raise ValueError(f"Node {node_id} needs numeric order")
        if not math.isfinite(order):
            raise ValueError(f"Node {node_id} has non-finite order")
        node_page = node.get("page_number")
        if node_page is not None and (
            isinstance(node_page, bool)
            or not isinstance(node_page, int)
            or node_page < 1
        ):
            raise ValueError(f"Node {node_id} has invalid page_number")
        if page_number is not None and node_page is not None and node_page != page_number:
            raise ValueError(
                f"Node {node_id} page_number does not match requested page_number"
            )
        _validate_layout(node_id, node.get("layout"))

    for node in nodes:
        parent_id = node.get("parent_id")
        if parent_id is not None and parent_id not in ids:
            raise ValueError(f"Node {node['id']} references unknown parent_id")
        for target_id in _reference_ids(node):
            if target_id not in ids:
                raise ValueError(f"Node {node['id']} references unknown node {target_id}")


def _validate_layout(node_id: str, layout: Any) -> None:
    if layout is None:
        return
    if not isinstance(layout, dict):
        raise ValueError(f"Node {node_id} layout must be an object")
    for field in ("x", "y", "width", "height"):
        value = layout.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Node {node_id} layout needs numeric {field}")
        if not math.isfinite(value):
            raise ValueError(f"Node {node_id} layout has non-finite {field}")
    if (
        layout["x"] < 0
        or layout["y"] < 0
        or layout["width"] <= 0
        or layout["height"] <= 0
        or layout["x"] + layout["width"] > 1.0
        or layout["y"] + layout["height"] > 1.0
    ):
        raise ValueError(f"Node {node_id} layout must be normalized within the page")


def _reference_ids(node: dict[str, Any]) -> list[str]:
    values = []
    for field in ("anchor_id", "target_id", "footnote_id", "comment_id"):
        if node.get(field) is not None:
            values.append(str(node[field]))
    references = node.get("references", [])
    if references is not None:
        if not isinstance(references, list) or not all(
            isinstance(reference, str) for reference in references
        ):
            raise ValueError(f"Node {node['id']} references must be an array of IDs")
        values.extend(references)
    return values


def _derive_evidence(graph: dict[str, Any]) -> dict[str, Any]:
    """Calculate structural and optional layout facts; do not infer visual meaning."""
    nodes = sorted(graph["nodes"], key=lambda node: (node["order"], node["id"]))
    by_id = {node["id"]: node for node in nodes}
    parent_edges = [
        {"parent": node["parent_id"], "child": node["id"]}
        for node in nodes
        if node.get("parent_id") is not None
    ]
    adjacency = [
        {"before": left["id"], "after": right["id"]}
        for left, right in zip(nodes, nodes[1:])
    ]
    list_groups = _list_groups(nodes)
    heading_scopes = _heading_scopes(nodes)
    anchors = [
        {"node": node["id"], "anchor": node["anchor_id"]}
        for node in nodes
        if node.get("anchor_id") is not None
    ]
    references = [
        {"node": node["id"], "target": target}
        for node in nodes
        for target in _reference_ids(node)
    ]
    return {
        "parent_edges": parent_edges,
        "document_order_adjacency": adjacency,
        "list_groups": list_groups,
        "heading_scopes": heading_scopes,
        "anchor_edges": anchors,
        "explicit_references": references,
        "layout_pairs": _layout_evidence(nodes, graph.get("page_number")),
        "node_count": len(nodes),
        "node_types": sorted({node["type"] for node in by_id.values()}),
    }


def _list_groups(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[str]] = {}
    for node in nodes:
        list_id = node.get("list_id", node.get("num_id"))
        if list_id is None:
            continue
        level = node.get("list_level", 0)
        if isinstance(level, bool) or not isinstance(level, int) or level < 0:
            continue
        groups.setdefault((str(list_id), level), []).append(node["id"])
    return [
        {"list_id": list_id, "level": level, "items": item_ids}
        for (list_id, level), item_ids in groups.items()
        if len(item_ids) >= 2
    ]


def _heading_scopes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    headings = [
        (index, node, _heading_level(node))
        for index, node in enumerate(nodes)
        if _heading_level(node) is not None
    ]
    scopes = []
    for position, (index, heading, level) in enumerate(headings):
        end = len(nodes)
        for next_index, _, next_level in headings[position + 1 :]:
            if next_level <= level:
                end = next_index
                break
        members = [node["id"] for node in nodes[index + 1 : end]]
        if members:
            scopes.append(
                {"heading": heading["id"], "level": level, "members": members}
            )
    return scopes


def _heading_level(node: dict[str, Any]) -> int | None:
    level = node.get("outline_level")
    if isinstance(level, int) and not isinstance(level, bool) and level >= 0:
        return level
    if str(node.get("type") or "").lower() == "heading":
        return 0
    return None


def _layout_evidence(
    nodes: list[dict[str, Any]], graph_page: int | None
) -> list[dict[str, Any]]:
    pairs = []
    positioned = [node for node in nodes if node.get("layout")]
    for left, right in combinations(positioned, 2):
        left_page = left.get("page_number", graph_page)
        right_page = right.get("page_number", graph_page)
        if left_page is None or left_page != right_page:
            continue
        left_layout, right_layout = left["layout"], right["layout"]
        kinds = []
        if abs(left_layout["x"] - right_layout["x"]) <= _ALIGN_TOLERANCE:
            kinds.append("left_aligned")
        if abs(left_layout["y"] - right_layout["y"]) <= _ALIGN_TOLERANCE:
            kinds.append("top_aligned")
        if _edge_distance(left_layout, right_layout) <= _PROXIMITY_TOLERANCE:
            kinds.append("proximate")
        if _contains(left_layout, right_layout):
            kinds.append("contains")
        elif _contains(right_layout, left_layout):
            pairs.append(
                {"kind": "contains", "container": right["id"], "member": left["id"]}
            )
        for kind in kinds:
            if kind != "contains":
                pairs.append({"kind": kind, "shapes": [left["id"], right["id"]]})
            elif _contains(left_layout, right_layout):
                pairs.append(
                    {"kind": "contains", "container": left["id"], "member": right["id"]}
                )
    return pairs


def _contains(container: dict[str, float], member: dict[str, float]) -> bool:
    return (
        container["x"] <= member["x"]
        and container["y"] <= member["y"]
        and container["x"] + container["width"] >= member["x"] + member["width"]
        and container["y"] + container["height"] >= member["y"] + member["height"]
        and container["width"] * container["height"]
        > member["width"] * member["height"] * 1.15
    )


def _edge_distance(left: dict[str, float], right: dict[str, float]) -> float:
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
        raise ValueError("DOCX structure manifest has no prompt files")
    s3 = boto3.client("s3")
    return "\n".join(
        s3.get_object(
            Bucket=bucket,
            Key=f"prompts/docx_structure_specialist/{filename}",
        )["Body"].read().decode("utf-8")
        for filename in files
    )


def _interpret(
    prompt: str,
    scope_id: str,
    page_number: int | None,
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
        f"Interpret DOCX scope {scope_id!r}. "
        f"Requested page number: {page_number if page_number is not None else 'none'}.\n"
        "Do not claim a relationship that lacks IDs in deterministic evidence.\n\n"
        f"DOCUMENT_STRUCTURE_GRAPH:\n{json.dumps(graph, ensure_ascii=False)}\n\n"
        f"DETERMINISTIC_EVIDENCE:\n{json.dumps(evidence, ensure_ascii=False)}"
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
    raise ValueError("Model returned no DOCX structure result")


def _save_evidence(
    output_bucket: str,
    session_id: str,
    source_document_path: str,
    scope_id: str,
    page_number: int | None,
    timestamp: str,
    graph: dict[str, Any],
    evidence: dict[str, Any],
) -> str:
    key = f"{session_id}/docx_structure/docx_structure_evidence_{scope_id}_{timestamp}.json"
    artifact = {
        "schema_version": "1.0",
        "analysis_mode": "native_ooxml_structure",
        "source_document_path": source_document_path,
        "scope_id": scope_id,
        "page_number": page_number,
        "document_structure_graph": graph,
        "deterministic_evidence": evidence,
    }
    boto3.client("s3").put_object(
        Bucket=output_bucket,
        Key=key,
        Body=json.dumps(artifact, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json",
        Tagging=f"session_id={session_id}&type=docx_structure_evidence",
    )
    return f"s3://{output_bucket}/{key}"


def _validate_model_result(
    result: str,
    scope_id: str,
    page_number: int | None,
    graph: dict[str, Any],
) -> None:
    try:
        root = ElementTree.fromstring(result)
    except ElementTree.ParseError as error:
        raise ValueError("Model result is not valid XML") from error
    if root.tag != "docx_structure_analysis":
        raise ValueError("Model result root must be docx_structure_analysis")
    if root.attrib.get("scope") != scope_id:
        raise ValueError("Model result scope does not match the request")
    if root.attrib.get("analysis_mode") != "native_ooxml_structure":
        raise ValueError("Model result must declare native_ooxml_structure mode")
    if page_number is not None and root.attrib.get("page") != str(page_number):
        raise ValueError("Model result page number does not match the request")
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

    node_ids = {node["id"] for node in graph["nodes"]}
    for relationship in root.findall("./relationships/relationship"):
        required = ("id", "type", "confidence", "source_node", "target_node")
        if any(not relationship.attrib.get(attribute) for attribute in required):
            raise ValueError(
                "Each relationship must have id, type, confidence, source_node, "
                "and target_node attributes"
            )
        source, target = relationship.attrib["source_node"], relationship.attrib["target_node"]
        if source not in node_ids or target not in node_ids:
            raise ValueError(
                "Each relationship must reference known source_node and target_node IDs"
            )
        if source == target:
            raise ValueError("A relationship must reference two distinct nodes")
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
                "Relationship evidence must cite both source_node and target_node IDs"
            )
