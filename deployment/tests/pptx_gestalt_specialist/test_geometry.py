import pytest


def _graph():
    return {
        "slide_number": 2,
        "shapes": [
            {"id": "container", "type": "rectangle", "x": 0.05, "y": 0.05, "width": 0.70, "height": 0.60},
            {"id": "title", "type": "text", "x": 0.10, "y": 0.10, "width": 0.40, "height": 0.08, "text": "Revenue"},
            {"id": "value", "type": "text", "x": 0.10, "y": 0.20, "width": 0.25, "height": 0.08, "text": "$10M"},
            {"id": "next", "type": "text", "x": 0.40, "y": 0.20, "width": 0.25, "height": 0.08, "text": "Next"},
            {
                "id": "arrow",
                "type": "connector",
                "x": 0.35,
                "y": 0.22,
                "width": 0.05,
                "height": 0.02,
                "from_shape": "value",
                "to_shape": "next",
                "direction": "right",
            },
        ],
    }


def test_derives_alignment_containment_and_connector_evidence(gestalt_module):
    evidence = gestalt_module._derive_evidence(_graph())

    assert {"kind": "left_aligned", "shapes": ["title", "value"]} in evidence["alignment_pairs"]
    assert {"container": "container", "member": "title"} in evidence["containment_pairs"]
    assert evidence["connector_edges"] == [
        {
            "connector": "arrow",
            "from_shape": "value",
            "to_shape": "next",
            "direction": "right",
        }
    ]


@pytest.mark.parametrize(
    "shape",
    [
        {"id": "off_slide", "x": 0.9, "y": 0.1, "width": 0.2, "height": 0.1},
        {"id": "zero_width", "x": 0.1, "y": 0.1, "width": 0.0, "height": 0.1},
        {"id": "nan", "x": float("nan"), "y": 0.1, "width": 0.1, "height": 0.1},
    ],
)
def test_rejects_non_normalized_or_invalid_bounds(gestalt_module, shape):
    with pytest.raises(ValueError):
        gestalt_module._validate_scene_graph({"slide_number": 2, "shapes": [shape]}, 2)


def test_rejects_model_relationship_to_unknown_shape(gestalt_module):
    result = """<pptx_gestalt_analysis slide="2" analysis_mode="native_coordinate_gestalt">
      <relationships><relationship id="r1" type="candidate_caption_for" confidence="low" source_shape="title" target_shape="missing">
        <evidence kind="proximity">title is near missing</evidence>
      </relationship></relationships>
      <uncertainties />
      <recommended_visual_analysis required="false">Not required.</recommended_visual_analysis>
    </pptx_gestalt_analysis>"""

    with pytest.raises(ValueError, match="known source_shape"):
        gestalt_module._validate_model_result(result, 2, _graph())


def test_accepts_additive_relationship_xml(gestalt_module):
    result = """<pptx_gestalt_analysis slide="2" analysis_mode="native_coordinate_gestalt">
      <relationships><relationship id="r1" type="connector_flow" confidence="high" source_shape="value" target_shape="next">
        <evidence kind="connector">value connects to next</evidence>
      </relationship></relationships>
      <uncertainties><uncertainty>Image content is not represented.</uncertainty></uncertainties>
      <recommended_visual_analysis required="false">Not required by this scene graph.</recommended_visual_analysis>
    </pptx_gestalt_analysis>"""

    gestalt_module._validate_model_result(result, 2, _graph())


def test_rejects_relationship_without_evidence(gestalt_module):
    result = """<pptx_gestalt_analysis slide="2" analysis_mode="native_coordinate_gestalt">
      <relationships><relationship id="r1" type="connector_flow" confidence="high" source_shape="value" target_shape="next" /></relationships>
      <uncertainties />
      <recommended_visual_analysis required="false">Not required.</recommended_visual_analysis>
    </pptx_gestalt_analysis>"""

    with pytest.raises(ValueError, match="deterministic evidence"):
        gestalt_module._validate_model_result(result, 2, _graph())


def test_rejects_evidence_that_does_not_cite_the_related_shapes(gestalt_module):
    result = """<pptx_gestalt_analysis slide="2" analysis_mode="native_coordinate_gestalt">
      <relationships><relationship id="r1" type="connector_flow" confidence="high" source_shape="value" target_shape="next">
        <evidence kind="connector">An arrow establishes the relationship.</evidence>
      </relationship></relationships>
      <uncertainties />
      <recommended_visual_analysis required="false">Not required.</recommended_visual_analysis>
    </pptx_gestalt_analysis>"""

    with pytest.raises(ValueError, match="cite both"):
        gestalt_module._validate_model_result(result, 2, _graph())
