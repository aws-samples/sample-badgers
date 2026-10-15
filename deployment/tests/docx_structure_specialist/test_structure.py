import pytest


def _graph():
    return {
        "page_number": 2,
        "nodes": [
            {
                "id": "heading-1",
                "type": "heading",
                "order": 1,
                "outline_level": 0,
                "text": "Market outlook",
                "page_number": 2,
                "layout": {"x": 0.1, "y": 0.1, "width": 0.5, "height": 0.06},
            },
            {
                "id": "paragraph-1",
                "type": "paragraph",
                "order": 2,
                "parent_id": "heading-1",
                "text": "Demand is rising.",
                "page_number": 2,
                "layout": {"x": 0.1, "y": 0.2, "width": 0.5, "height": 0.08},
            },
            {
                "id": "list-1",
                "type": "list_item",
                "order": 3,
                "list_id": "list-42",
                "list_level": 0,
                "text": "North America",
                "page_number": 2,
            },
            {
                "id": "list-2",
                "type": "list_item",
                "order": 4,
                "list_id": "list-42",
                "list_level": 0,
                "text": "Europe",
                "page_number": 2,
            },
            {
                "id": "chart-1",
                "type": "drawing",
                "order": 5,
                "anchor_id": "paragraph-1",
                "page_number": 2,
            },
            {
                "id": "reference-1",
                "type": "paragraph",
                "order": 6,
                "target_id": "chart-1",
                "text": "See chart.",
                "page_number": 2,
            },
        ],
    }


def test_derives_native_hierarchy_list_anchor_and_reference_evidence(structure_module):
    evidence = structure_module._derive_evidence(_graph())

    assert {"parent": "heading-1", "child": "paragraph-1"} in evidence["parent_edges"]
    assert evidence["list_groups"] == [
        {"list_id": "list-42", "level": 0, "items": ["list-1", "list-2"]}
    ]
    assert {"node": "chart-1", "anchor": "paragraph-1"} in evidence["anchor_edges"]
    assert {"node": "reference-1", "target": "chart-1"} in evidence["explicit_references"]
    assert evidence["heading_scopes"] == [
        {
            "heading": "heading-1",
            "level": 0,
            "members": ["paragraph-1", "list-1", "list-2", "chart-1", "reference-1"],
        }
    ]


def test_does_not_compare_layout_across_unknown_pages(structure_module):
    graph = _graph()
    del graph["page_number"]
    for node in graph["nodes"]:
        node.pop("page_number", None)

    assert structure_module._derive_evidence(graph)["layout_pairs"] == []


def test_rejects_unknown_native_reference(structure_module):
    graph = _graph()
    graph["nodes"][-1]["target_id"] = "unknown-node"

    with pytest.raises(ValueError, match="unknown node"):
        structure_module._validate_graph(graph, 2)


def test_rejects_layout_outside_normalized_page(structure_module):
    graph = _graph()
    graph["nodes"][0]["layout"]["width"] = 1.0

    with pytest.raises(ValueError, match="normalized"):
        structure_module._validate_graph(graph, 2)


def test_accepts_evidence_backed_additive_result(structure_module):
    result = """<docx_structure_analysis scope="page-2" page="2" analysis_mode="native_ooxml_structure">
      <relationships><relationship id="r1" type="anchored_to" confidence="high" source_node="chart-1" target_node="paragraph-1">
        <evidence kind="anchor">chart-1 has anchor paragraph-1</evidence>
      </relationship></relationships>
      <uncertainties><uncertainty>Chart contents require pixels.</uncertainty></uncertainties>
      <recommended_visual_analysis required="true">Chart meaning needs visual analysis.</recommended_visual_analysis>
    </docx_structure_analysis>"""

    structure_module._validate_model_result(result, "page-2", 2, _graph())


def test_rejects_result_without_node_ids_in_evidence(structure_module):
    result = """<docx_structure_analysis scope="page-2" page="2" analysis_mode="native_ooxml_structure">
      <relationships><relationship id="r1" type="anchored_to" confidence="high" source_node="chart-1" target_node="paragraph-1">
        <evidence kind="anchor">The drawing is anchored to its paragraph.</evidence>
      </relationship></relationships>
      <uncertainties />
      <recommended_visual_analysis required="false">Not required.</recommended_visual_analysis>
    </docx_structure_analysis>"""

    with pytest.raises(ValueError, match="cite both"):
        structure_module._validate_model_result(result, "page-2", 2, _graph())
