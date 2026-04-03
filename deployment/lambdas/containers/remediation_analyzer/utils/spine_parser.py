"""
CHECKPOINT_03C_PARSER
=====================
Spine parser — reads unified_document XML and builds a StructureElement tree.

Handles two schema versions:
  v1.0 (flat): <content_spine> with flat <element> siblings → infer hierarchy
  v2.0 (nested): <content_tree> with <sect> nesting → direct tree build

Used by lambda_handler.py to replace the old _parse_correlation_xml.
"""

import logging
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

from utils.pdf_accessibility_models import (
    StructureElement,
    ElementRole,
    TagRegion,
    get_role,
    is_heading,
    heading_level,
    infer_structure_from_flat_regions,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Public API
# ============================================================================


def parse_correlation_xml(
    xml_content: str,
) -> Tuple[Optional[StructureElement], Dict[int, List[Dict[str, Any]]]]:
    """Parse correlation XML into a StructureElement tree and flat element list.

    Returns:
        (structure_tree, page_elements)
        - structure_tree: StructureElement root (Document), or None if parse fails
        - page_elements: dict mapping page_num (1-indexed) to list of element dicts
          compatible with the existing lambda_handler pipeline (id, type, order,
          text, alt_text, caption). This ensures bbox resolution still works.
    """
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        logger.warning("Failed to parse correlation XML: %s", e)
        return None, {}

    schema_version = root.get("schema_version", "1.0")

    if schema_version == "2.0":
        return _parse_v2(root)
    else:
        return _parse_v1(root)


# ============================================================================
# V2.0 Parser — nested <content_tree> with <sect> hierarchy
# ============================================================================


def _parse_v2(
    root: ET.Element,
) -> Tuple[Optional[StructureElement], Dict[int, List[Dict[str, Any]]]]:
    """Parse schema v2.0 — hierarchical content_tree."""

    page_num = int(root.get("page", "1"))

    content_tree = root.find(".//content_tree")
    if content_tree is None:
        logger.warning("No <content_tree> found in v2.0 document")
        return None, {}

    # Build the StructureElement tree
    doc = StructureElement(id="doc", tag="Document", role=ElementRole.GROUPING, depth=0)

    # Collect flat element dicts for bbox resolution pipeline
    flat_elements: List[Dict[str, Any]] = []

    # Process children of content_tree (should be a single sect_root)
    for child in content_tree:
        _parse_node_v2(child, doc, page_num, flat_elements)

    page_elements = {page_num: flat_elements} if flat_elements else {}

    logger.info(
        "Parsed v2.0: %d tree nodes, %d flat elements for page %d",
        sum(1 for _ in doc.walk()),
        len(flat_elements),
        page_num,
    )

    return doc, page_elements


def _parse_node_v2(
    xml_node: ET.Element,
    parent: StructureElement,
    default_page: int,
    flat_elements: List[Dict[str, Any]],
) -> None:
    """Recursively parse an XML node into the StructureElement tree.

    Handles:
      - <sect> → Sect grouping element
      - <element> → content element (may have children for table/list substructure)
      - <inline> children inside <element>
    """
    tag_name = xml_node.tag

    if tag_name == "sect":
        sect_id = xml_node.get("id", f"sect_auto_{id(xml_node)}")
        # Check if sect has an explicit tag override (e.g. BlockQuote)
        explicit_tag = xml_node.get("tag", "Sect")

        sect_elem = parent.add_child(
            StructureElement(
                id=sect_id,
                tag=explicit_tag,
                role=ElementRole.GROUPING,
            )
        )

        for child in xml_node:
            _parse_node_v2(child, sect_elem, default_page, flat_elements)

    elif tag_name == "element":
        _parse_element_v2(xml_node, parent, default_page, flat_elements)

    # Skip metadata, enrichments, cross_reference_index etc. at this level


def _parse_element_v2(
    xml_node: ET.Element,
    parent: StructureElement,
    default_page: int,
    flat_elements: List[Dict[str, Any]],
) -> None:
    """Parse a single <element> node into a StructureElement."""

    elem_id = xml_node.get("id", "")
    elem_tag = xml_node.get("tag", "P")
    elem_order = int(xml_node.get("order", "0"))
    elem_page = int(xml_node.get("page", str(default_page)))
    elem_scope = xml_node.get("scope")
    elem_ref = xml_node.get("ref")
    elem_parent_ref = xml_node.get("parent_ref")
    elem_placement = xml_node.get("placement")
    elem_lang = xml_node.get("lang", "")

    # Extract text
    text_node = xml_node.find("text")
    text = text_node.text.strip() if text_node is not None and text_node.text else ""

    # Extract alt_text
    alt_node = xml_node.find("alt_text")
    alt_text = alt_node.text.strip() if alt_node is not None and alt_node.text else ""

    # Extract caption (for figures from v1 compat)
    caption_node = xml_node.find("caption")
    caption = (
        caption_node.text.strip()
        if caption_node is not None and caption_node.text
        else ""
    )

    # Build the StructureElement
    role = get_role(elem_tag)
    se = StructureElement(
        id=elem_id,
        tag=elem_tag,
        role=role,
        order=elem_order,
        page=elem_page,
        text=text,
        alt_text=alt_text,
        scope=elem_scope,
        ref_id=elem_ref,
        placement=elem_placement,
    )

    parent.add_child(se)

    # Process inline children
    inline_node = xml_node.find("inline")
    if inline_node is not None:
        for inline_elem in inline_node.findall("element"):
            _parse_inline_v2(inline_elem, se)

    # Process structural children (for tables, lists)
    children_node = xml_node.find("children")
    if children_node is not None:
        for child_elem in children_node:
            if child_elem.tag == "element":
                _parse_element_v2(child_elem, se, default_page, flat_elements)

    # ── Extract enrichments (generic across all analyzer types) ──
    enrichment = _extract_enrichments(xml_node.find("enrichments"))

    # Add to flat elements list for bbox resolution
    # Only add leaf-level content elements (not structural wrappers like THead, TR)
    if _is_bbox_resolvable(elem_tag):
        flat_entry = {
            "id": elem_id,
            "type": _normalize_type_for_resolver(elem_tag),
            "tag": elem_tag,  # preserve original PDF/UA tag for structure tree
            "order": elem_order,
            "text": text,
            "alt_text": enrichment["description"] or alt_text,
            "caption": caption or (text if elem_tag == "Caption" else ""),
            "figure_title": enrichment["title"],
            "actual_text": enrichment["actual_text"],
            "language": elem_lang or None,
            "enrichment_title": enrichment["title"],
            "enrichment_description": enrichment["description"],
            "enrichment_actual_text": enrichment["actual_text"],
            "enrichment_tags": enrichment["tags"],
            "enrichment_related": enrichment["related"],
        }
        flat_elements.append(flat_entry)


def _extract_enrichments(enrichments_node) -> Dict[str, Any]:
    """Extract accessibility-relevant data from enrichments (badgers schema v1.0).

    Returns:
        title: str — for /T on the StructElem
        description: str — for /Alt (figures) or extended description
        actual_text: str — for /ActualText structured summary
        tags: list[str] — semantic tags from keyword_topic
        related: list[dict] — [{element_id, relationship, evidence}]
    """
    result: Dict[str, Any] = {
        "title": "",
        "description": "",
        "actual_text": "",
        "tags": [],
        "related": [],
    }

    if enrichments_node is None:
        return result

    # ── cross_reference ──
    for xref in enrichments_node.findall("cross_reference"):
        rel_node = xref.find("relationship")
        ev_node = xref.find("evidence")
        result["related"].append({
            "element_id": xref.get("target", ""),
            "relationship": rel_node.text.strip() if rel_node is not None and rel_node.text else xref.get("type", ""),
            "evidence": ev_node.text.strip() if ev_node is not None and ev_node.text else "",
        })

    # ── keyword_topic ──
    kt = enrichments_node.find("keyword_topic")
    if kt is not None:
        for kw in kt.findall(".//keyword"):
            term = kw.get("term", "")
            if term:
                result["tags"].append(term)
        for topic in kt.findall(".//topic"):
            name = topic.get("name", "")
            if name:
                result["tags"].append(name)

    # ── chart (Part 6) ──
    chart = enrichments_node.find("chart")
    if chart is not None:
        _extract_text(chart, "title", result, "title")
        _extract_text(chart, "description", result, "description")
        _extract_text(chart, "context", result, "description")
        series = []
        for dp in chart.findall(".//data_points/point"):
            x_node = dp.find("x")
            y_node = dp.find("y")
            if x_node is not None and x_node.text and y_node is not None and y_node.text:
                series.append(f"{x_node.text.strip()}: {y_node.text.strip()}")
        if series:
            result["actual_text"] = "; ".join(series)
        _extract_figure_reference(chart, result)

    # ── table (Part 7) ──
    tbl = enrichments_node.find("table")
    if tbl is not None:
        _extract_text(tbl, "title", result, "title")
        _extract_text(tbl, "description", result, "description")
        _extract_text(tbl, "context", result, "description")
        _extract_figure_reference(tbl, result)

    # ── diagram (Part 8) ──
    diag = enrichments_node.find("diagram")
    if diag is not None:
        _extract_text(diag, "title", result, "title")
        _extract_text(diag, "description", result, "description")
        _extract_text(diag, "flow_description", result, "description")
        _extract_text(diag, "context", result, "description")
        parts = []
        for node in diag.findall(".//nodes/node"):
            label_n = node.find("label")
            label = label_n.text.strip() if label_n is not None and label_n.text else ""
            ntype = node.get("type", "")
            if label:
                parts.append(f"{label} ({ntype})" if ntype else label)
        conns = []
        for edge in diag.findall(".//edges/edge"):
            fr, to = edge.get("from", ""), edge.get("to", "")
            label_n = edge.find("label")
            label = label_n.text.strip() if label_n is not None and label_n.text else ""
            if fr and to:
                conns.append(f"{fr} -> {to}" + (f" [{label}]" if label else ""))
        if parts:
            result["actual_text"] = "Nodes: " + ", ".join(parts)
        if conns:
            result["actual_text"] += ". Connections: " + "; ".join(conns)
        _extract_figure_reference(diag, result)

    # ── decision_tree ──
    dt = enrichments_node.find("decision_tree")
    if dt is not None:
        _extract_text(dt, "title", result, "title")
        _extract_text(dt, "purpose", result, "description")
        outcomes = []
        for outcome in dt.findall(".//outcomes/outcome"):
            res = outcome.get("result", "")
            path = outcome.get("path", "")
            if res:
                outcomes.append(f"{res} (path: {path})" if path else res)
        if outcomes:
            result["actual_text"] = "Outcomes: " + "; ".join(outcomes)

    # ── code_block (Part 11) ──
    cb = enrichments_node.find("code_block")
    if cb is not None:
        _extract_text(cb, "context", result, "description")
        lang_n = cb.find("language")
        if lang_n is not None and lang_n.text:
            result["title"] = f"Code ({lang_n.text.strip()})"
        code_n = cb.find("code_text")
        if code_n is not None and code_n.text and not result["actual_text"]:
            result["actual_text"] = code_n.text.strip()

    # ── handwriting (Part 9) ──
    hw = enrichments_node.find("handwriting")
    if hw is not None:
        _extract_text(hw, "context", result, "description")
        segments = []
        for seg in hw.findall(".//transcription/segment"):
            text_n = seg.find("text")
            if text_n is not None and text_n.text:
                segments.append(text_n.text.strip())
        if segments and not result["actual_text"]:
            result["actual_text"] = " ".join(segments)

    # ── handwriting_math ──
    hwm = enrichments_node.find("handwriting_math")
    if hwm is not None:
        for chain in hwm.findall("solution_chain"):
            _extract_text(chain, "correctness_notes", result, "description")
            fa = chain.find("final_answer/latex")
            if fa is not None and fa.text:
                result["actual_text"] = fa.text.strip()
            prob_ref = chain.get("problem_ref", "")
            ans_ref = chain.get("answer_ref", "")
            if prob_ref:
                result["related"].append({"element_id": prob_ref, "relationship": "problem_ref", "evidence": ""})
            if ans_ref:
                result["related"].append({"element_id": ans_ref, "relationship": "answer_ref", "evidence": ""})

    # ── war_map ──
    wm = enrichments_node.find("war_map")
    if wm is not None:
        _extract_text(wm, "title", result, "title")
        parts = []
        for feature in wm.findall(".//terrain/feature"):
            name_n = feature.find("name")
            sig_n = feature.find("tactical_significance")
            if name_n is not None and name_n.text:
                part = name_n.text.strip()
                if sig_n is not None and sig_n.text:
                    part += f": {sig_n.text.strip()}"
                parts.append(part)
        if parts and not result["description"]:
            result["description"] = "; ".join(parts)
        forces = []
        for force in wm.findall(".//military_forces/force"):
            aff = force.get("affiliation", "")
            units = [u.find("designation").text.strip() for u in force.findall("unit")
                     if u.find("designation") is not None and u.find("designation").text]
            if aff and units:
                forces.append(f"{aff}: {', '.join(units)}")
        if forces and not result["actual_text"]:
            result["actual_text"] = "; ".join(forces)

    return result


def _extract_text(parent_node, child_tag: str, result: dict, result_key: str):
    """Helper: extract text from a child element into result[result_key] if not already set."""
    if result.get(result_key):
        return
    node = parent_node.find(child_tag)
    if node is not None and node.text:
        result[result_key] = node.text.strip()


def _extract_figure_reference(specialized_node, result: dict):
    """Extract figure_reference block into related array."""
    fig_ref = specialized_node.find("figure_reference")
    if fig_ref is None:
        return
    ref_text_n = fig_ref.find("referencing_text")
    caption_n = fig_ref.find("caption")
    fig_id_n = fig_ref.find("figure_id") or fig_ref.find("table_id")
    if ref_text_n is not None and ref_text_n.text:
        result["related"].append({
            "element_id": fig_id_n.text.strip() if fig_id_n is not None and fig_id_n.text else "",
            "relationship": "referenced_by_text",
            "evidence": ref_text_n.text.strip(),
        })
    if caption_n is not None and caption_n.text and not result.get("title"):
        result["title"] = caption_n.text.strip()


def _parse_inline_v2(
    xml_node: ET.Element,
    parent: StructureElement,
) -> None:
    """Parse an inline element within a parent element."""

    elem_id = xml_node.get("id", "")
    elem_tag = xml_node.get("tag", "Span")
    elem_ref = xml_node.get("ref")
    elem_placement = xml_node.get("placement", "Inline")

    text_node = xml_node.find("text")
    text = text_node.text.strip() if text_node is not None and text_node.text else ""

    se = StructureElement(
        id=elem_id,
        tag=elem_tag,
        role=ElementRole.INLINE,
        text=text,
        ref_id=elem_ref,
        placement=elem_placement,
    )

    parent.add_inline(se)


# ============================================================================
# V1.0 Parser — flat <content_spine> (backwards compat)
# ============================================================================


def _parse_v1(
    root: ET.Element,
) -> Tuple[Optional[StructureElement], Dict[int, List[Dict[str, Any]]]]:
    """Parse schema v1.0 — flat content_spine.

    Extracts flat elements, builds flat element list for bbox resolution,
    then infers hierarchy from heading levels using infer_structure_from_flat_regions.
    """
    page_num = int(root.get("page", "1"))
    flat_elements: List[Dict[str, Any]] = []

    # Also build TagRegions for hierarchy inference
    tag_regions: List[TagRegion] = []

    for elem in root.iter("element"):
        elem_id = elem.get("id", "")
        elem_type = elem.get("type", "paragraph")
        elem_order = int(elem.get("order", "0"))

        text_node = elem.find("text")
        text = (
            text_node.text.strip() if text_node is not None and text_node.text else ""
        )

        alt_node = elem.find("alt_text")
        alt_text = (
            alt_node.text.strip() if alt_node is not None and alt_node.text else ""
        )

        caption_node = elem.find("caption")
        caption = (
            caption_node.text.strip()
            if caption_node is not None and caption_node.text
            else ""
        )

        flat_elements.append(
            {
                "id": elem_id,
                "type": elem_type,
                "order": elem_order,
                "text": text,
                "alt_text": alt_text,
                "caption": caption,
            }
        )

        # Build TagRegion for hierarchy inference
        tag_regions.append(
            TagRegion(
                tag=elem_type,
                bbox=(0, 0, 0, 0),  # No bbox yet — resolved later
                text_content=text,
                alt_text=alt_text,
                order=elem_order,
                page=page_num - 1,  # TagRegion uses 0-indexed pages
                element_id=elem_id,
            )
        )

    page_elements = {page_num: flat_elements} if flat_elements else {}

    # Infer hierarchy from heading levels
    if tag_regions:
        regions_by_page = {page_num - 1: tag_regions}  # 0-indexed
        tree = infer_structure_from_flat_regions(regions_by_page)
    else:
        tree = None

    logger.info(
        "Parsed v1.0 (flat): %d elements for page %d, hierarchy %s",
        len(flat_elements),
        page_num,
        "inferred" if tree else "unavailable",
    )

    return tree, page_elements


# ============================================================================
# Helper functions
# ============================================================================


def _is_bbox_resolvable(tag: str) -> bool:
    """Should this element be sent to the bbox resolver?

    Structural wrapper tags (THead, TBody, TR, LI, Lbl, LBody) don't need
    their own bboxes — they inherit from their content children.
    We resolve bboxes for content-bearing leaves.
    """
    non_resolvable = {
        "Sect",
        "THead",
        "TBody",
        "TFoot",
        "TR",
        "LI",
        "LBody",
        "TOC",
        "TOCI",
        "Document",
        "Part",
        "Art",
        "Div",
    }
    return tag not in non_resolvable


def _normalize_type_for_resolver(tag: str) -> str:
    """Normalize PDF/UA tag names to the types the resolver expects.

    The existing resolver pipeline uses lowercase types from the full_text
    analyzer (e.g., "paragraph", "h1", "figure"). This maps PDF/UA tags
    back to those names for compatibility.
    """
    mapping = {
        "H1": "h1",
        "H2": "h2",
        "H3": "h3",
        "H4": "h4",
        "H5": "h5",
        "H6": "h6",
        "P": "paragraph",
        "Figure": "figure",
        "Table": "table",
        "Caption": "caption",
        "L": "list",
        "Formula": "formula",
        "Note": "footnote",
        "BibEntry": "paragraph",
        "BlockQuote": "paragraph",
        "Code": "paragraph",
        "Artifact": "footer",
        "Lbl": "lbl",
        "TH": "paragraph",
        "TD": "paragraph",
    }
    return mapping.get(tag, "paragraph")


# ============================================================================
# Integration test
# ============================================================================


def _self_test():
    """Verify parser works against the example spine."""

    v2_xml = """
    <unified_document session_id="test" page="1" schema_version="2.0">
        <metadata>
            <source_image>test.png</source_image>
            <correlation_timestamp>2026-01-01T00:00:00Z</correlation_timestamp>
            <analyzers_executed>
                <analyzer name="full_text" s3_uri="s3://test/ft.xml"/>
            </analyzers_executed>
        </metadata>
        <content_tree>
            <sect id="sect_root">
                <element id="elem_001" tag="H1" order="1" page="1">
                    <text>Document Title</text>
                </element>
                <element id="elem_002" tag="P" order="2" page="1">
                    <text>Introduction paragraph with a formula v = d/t inside it.</text>
                    <inline>
                        <element id="elem_002_f1" tag="Formula" placement="Inline">
                            <text>v = d/t</text>
                        </element>
                    </inline>
                </element>
                <sect id="sect_methods">
                    <element id="elem_003" tag="H2" order="3" page="1">
                        <text>Methods</text>
                    </element>
                    <element id="elem_004" tag="P" order="4" page="1">
                        <text>We used the following approach.</text>
                    </element>
                    <element id="elem_005" tag="Figure" order="5" page="1">
                        <alt_text>A bar chart showing results.</alt_text>
                    </element>
                    <element id="elem_005_cap" tag="Caption" order="6" page="1" parent_ref="elem_005">
                        <text>Figure 1: Results overview</text>
                    </element>
                    <element id="table_001" tag="Table" order="7" page="1">
                        <children>
                            <element id="thead_001" tag="THead">
                                <children>
                                    <element id="tr_h" tag="TR">
                                        <children>
                                            <element id="th_1" tag="TH" scope="Column">
                                                <text>Variable</text>
                                            </element>
                                            <element id="th_2" tag="TH" scope="Column">
                                                <text>Value</text>
                                            </element>
                                        </children>
                                    </element>
                                </children>
                            </element>
                            <element id="tbody_001" tag="TBody">
                                <children>
                                    <element id="tr_1" tag="TR">
                                        <children>
                                            <element id="td_1" tag="TD">
                                                <text>Speed</text>
                                            </element>
                                            <element id="td_2" tag="TD">
                                                <text>42</text>
                                            </element>
                                        </children>
                                    </element>
                                </children>
                            </element>
                        </children>
                    </element>
                </sect>
                <element id="artifact_001" tag="Artifact" order="99" page="1">
                    <text>Page 1</text>
                </element>
            </sect>
        </content_tree>
    </unified_document>
    """

    tree, page_elements = parse_correlation_xml(v2_xml)

    assert tree is not None, "Tree should not be None"
    assert tree.tag == "Document", f"Root should be Document, got {tree.tag}"

    # Count nodes
    total = sum(1 for _ in tree.walk())
    print(f"Total tree nodes: {total}")

    # Verify nesting
    sect_root = tree.children[0]
    assert sect_root.tag == "Sect", f"First child should be Sect, got {sect_root.tag}"

    # Find the H2 section
    h2_sect = None
    for child in sect_root.children:
        if child.tag == "Sect" and child.id == "sect_methods":
            h2_sect = child
            break
    assert h2_sect is not None, "Should find sect_methods"

    # Verify H2 is inside sect_methods
    h2 = h2_sect.children[0]
    assert h2.tag == "H2", f"First child of sect_methods should be H2, got {h2.tag}"
    assert h2.text == "Methods"

    # Verify inline formula in elem_002
    elem_002 = sect_root.children[1]  # P after H1
    assert len(elem_002.inline_children) == 1, "P should have 1 inline child"
    assert elem_002.inline_children[0].tag == "Formula"
    assert elem_002.inline_children[0].text == "v = d/t"

    # Verify table substructure
    table = None
    for child in h2_sect.children:
        if child.tag == "Table":
            table = child
            break
    assert table is not None, "Should find Table in sect_methods"
    assert len(table.children) == 2, "Table should have THead + TBody"
    assert table.children[0].tag == "THead"
    assert table.children[1].tag == "TBody"

    # Verify TH has scope
    thead = table.children[0]
    tr = thead.children[0]
    th = tr.children[0]
    assert th.scope == "Column", f"TH scope should be Column, got {th.scope}"

    # Verify flat elements for bbox resolution
    assert 1 in page_elements, "Should have elements for page 1"
    flat = page_elements[1]
    print(f"Flat elements for bbox resolution: {len(flat)}")
    for e in flat:
        print(f"  {e['id']}: {e['type']} — {e['text'][:40]}")

    # Verify Artifact is in flat list (for content stream marking)
    artifact_ids = [e["id"] for e in flat if e["type"] == "footer"]
    assert "artifact_001" in artifact_ids, "Artifact should be in flat list"

    # --- V1 backwards compat test ---
    v1_xml = """
    <unified_document session_id="test" page="1">
        <content_spine>
            <element id="elem_001" type="h1" order="1">
                <text>Title</text>
            </element>
            <element id="elem_002" type="paragraph" order="2">
                <text>Body text</text>
            </element>
            <element id="elem_003" type="h2" order="3">
                <text>Section</text>
            </element>
            <element id="elem_004" type="paragraph" order="4">
                <text>Section body</text>
            </element>
        </content_spine>
    </unified_document>
    """

    tree_v1, pages_v1 = parse_correlation_xml(v1_xml)
    assert tree_v1 is not None, "V1 tree should not be None"
    assert tree_v1.tag == "Document"
    total_v1 = sum(1 for _ in tree_v1.walk())
    print(f"\nV1 backwards compat: {total_v1} tree nodes")

    # Verify V1 inferred hierarchy
    def show(elem, indent=0):
        print("  " * indent + f"<{elem.tag}> {elem.text[:30] if elem.text else ''}")
        for c in elem.children:
            show(c, indent + 1)

    show(tree_v1)

    print("\nAll parser tests passed ✓")


if __name__ == "__main__":
    _self_test()
