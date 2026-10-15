"""Unit tests for native DOCX parsing, schema validation, and page routing."""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pytest

PROCESSOR_DIR = (
    Path(__file__).resolve().parents[2] / "lambdas" / "containers" / "docx_processor"
)
sys.path.insert(0, str(PROCESSOR_DIR))

from docx_processor import (  # noqa: E402
    DocxProcessingError,
    NativeBlock,
    PageLayout,
    build_content_tree_xml,
    build_native_relationships,
    create_page_plan,
    decide_page_route,
    align_blocks_to_pages,
    inspect_docx,
    validate_content_tree,
)

SCHEMA_DIRECTORY = Path(__file__).resolve().parents[3] / "schemas"


def _docx(
    body: str,
    relationships: str = "",
    extra_parts: dict[str, bytes] | None = None,
) -> bytes:
    """Build the minimal OOXML package needed to exercise the low-level parser."""
    document = f"""<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
            xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
            xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
            xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">
  <w:body>{body}</w:body>
</w:document>"""
    content_types = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>"""
    with io.BytesIO() as stream:
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("[Content_Types].xml", content_types)
            archive.writestr("word/document.xml", document)
            if relationships:
                archive.writestr("word/_rels/document.xml.rels", relationships)
            for name, content in (extra_parts or {}).items():
                archive.writestr(name, content)
        return stream.getvalue()


def test_inspect_docx_extracts_heading_list_table_and_drawing() -> None:
    payload = _docx(
        """
        <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Introduction</w:t></w:r></w:p>
        <w:p><w:pPr><w:numPr><w:ilvl w:val="2"/><w:numId w:val="4"/></w:numPr></w:pPr><w:r><w:t>First item</w:t></w:r></w:p>
        <w:tbl>
          <w:tr><w:tc><w:p><w:r><w:t>Header</w:t></w:r></w:p></w:tc></w:tr>
          <w:tr><w:tc><w:tcPr><w:gridSpan w:val="2"/></w:tcPr><w:p><w:r><w:t>Cell</w:t></w:r></w:p></w:tc></w:tr>
        </w:tbl>
        <w:p>
          <w:r><w:drawing><wp:anchor><wp:docPr descr="Architecture diagram"/></wp:anchor></w:drawing></w:r>
          <w:r><w:t>Diagram caption</w:t></w:r>
        </w:p>
        """
    )

    blocks, metadata = inspect_docx(payload)

    assert [block.kind for block in blocks] == ["heading", "list", "table", "paragraph"]
    assert blocks[0].level == 1
    assert blocks[1].list_id == "4"
    assert blocks[1].list_level == 2
    assert blocks[2].has_merged_cells is True
    assert blocks[2].table_cells[0][0]["text"] == "Header"
    assert blocks[2].table_cells[1][0]["column_span"] == 2
    assert blocks[3].has_floating_drawing is True
    assert blocks[3].image_descriptions == ["Architecture diagram"]
    assert metadata["source_format"] == "docx"


def test_native_content_tree_is_valid_badgers_xml() -> None:
    xml_payload = build_content_tree_xml(
        1,
        [
            NativeBlock("word/document.xml:block-1", "heading", "A title", level=1),
            NativeBlock("word/document.xml:block-2", "paragraph", "A paragraph."),
            NativeBlock(
                "word/document.xml:block-3",
                "table",
                "A B",
                table_rows=1,
                table_cells=[[{"text": "A"}, {"text": "B"}]],
            ),
        ],
        "test-session",
    )

    valid, errors = validate_content_tree(xml_payload, SCHEMA_DIRECTORY)

    assert valid is True
    assert errors == []
    assert b'tag="TR"' in xml_payload
    assert b'tag="TH"' in xml_payload


def test_page_route_is_independent_and_explainable() -> None:
    simple_page = PageLayout(1, "Plain text", 0, 0, 612, 792)
    visual_page = PageLayout(2, "Diagram caption", 1, 12, 612, 792)
    simple_block = NativeBlock(
        "one", "paragraph", "Plain text", page_numbers=[1], alignment_confidence=0.98
    )
    visual_block = NativeBlock(
        "two",
        "paragraph",
        "Diagram caption",
        has_drawing=True,
        has_floating_drawing=True,
        images=[{"display_area_ratio": 0.4, "graphic_type": ""}],
        image_descriptions=["Architecture diagram"],
        page_numbers=[2],
        alignment_confidence=0.7,
    )

    simple_score, simple_route, simple_reasons = decide_page_route(
        simple_page, [simple_block], 1
    )
    visual_score, visual_route, visual_reasons = decide_page_route(
        visual_page,
        [visual_block],
        1,
        [
            {
                "type": "captioned_by",
                "source_id": "two",
                "target_id": "caption",
            }
        ],
    )

    assert (simple_score, simple_route, simple_reasons) == (
        0,
        "native_docx",
        ["aligned_native_text"],
    )
    assert visual_score >= 60
    assert visual_route == "vision_ai"
    assert "floating_or_wrapped_visual" in visual_reasons


def test_text_only_columns_and_alignment_uncertainty_do_not_force_vision() -> None:
    page = PageLayout(1, "Two-column text", 0, 0, 612, 792)
    aligned_block = NativeBlock(
        "one",
        "paragraph",
        "Two-column text",
        column_count=2,
        alignment_confidence=0.98,
    )
    uncertain_block = NativeBlock(
        "two",
        "paragraph",
        "Uncertainly aligned text",
        alignment_confidence=0.35,
    )

    column_score, column_route, column_reasons = decide_page_route(
        page, [aligned_block], 2
    )
    uncertain_score, uncertain_route, uncertain_reasons = decide_page_route(
        page, [uncertain_block], 1
    )

    assert (column_score, column_route, column_reasons) == (
        5,
        "native_docx",
        ["multi_column_layout"],
    )
    assert (uncertain_score, uncertain_route) == (20, "native_docx")
    assert uncertain_reasons == ["low_alignment_confidence"]


def test_complex_native_table_requires_review_but_not_full_vision() -> None:
    page = PageLayout(1, "Wide table", 0, 0, 612, 792)
    table = NativeBlock(
        "table",
        "table",
        "Wide table",
        table_rows=30,
        table_columns=12,
        has_repeated_header=True,
        alignment_confidence=0.98,
    )

    score, route, reasons = decide_page_route(page, [table], 1)

    assert (score, route) == (35, "native_docx")
    assert reasons == ["highly_complex_table"]


def test_captioned_photo_does_not_become_vision_without_semantic_visual_evidence() -> None:
    page = PageLayout(1, "Store photograph", 1, 0, 612, 792)
    photograph = NativeBlock(
        "photo",
        "paragraph",
        has_drawing=True,
        images=[{"display_area_ratio": 0.4, "graphic_type": ""}],
        image_descriptions=["Storefront photograph"],
        alignment_confidence=0.98,
    )
    relationships = [
        {
            "type": "captioned_by",
            "source_id": "photo",
            "target_id": "caption",
            "evidence": "Figure 1 — Storefront photograph",
        }
    ]

    score, route, reasons = decide_page_route(
        page, [photograph], 1, relationships
    )

    assert (score, route) == (50, "hybrid")
    assert "referenced_embedded_visual" in reasons
    assert "semantic_embedded_visual" not in reasons


def test_image_only_page_routes_to_textract_before_vision() -> None:
    page = PageLayout(3, "", 1, 0, 612, 792)
    image_only_block = NativeBlock(
        "three",
        "paragraph",
        has_drawing=True,
        page_numbers=[3],
        alignment_confidence=1.0,
    )

    score, route, reasons = decide_page_route(page, [image_only_block], 1)

    assert score == 15
    assert route == "textract_ocr"
    assert reasons == [
        "image_only_page_without_native_text",
        "textract_ocr_first_pass",
    ]


def test_explicit_page_break_advances_alignment_without_lowering_confidence() -> None:
    first = NativeBlock("one", "paragraph", "First page")
    page_break = NativeBlock("break", "page_break", has_page_break=True)
    second = NativeBlock("two", "paragraph", "Second page")
    pages = [
        PageLayout(1, "First page", 0, 0, 612, 792),
        PageLayout(2, "Second page", 0, 0, 612, 792),
    ]

    align_blocks_to_pages([first, page_break, second], pages)

    assert first.page_numbers == [1]
    assert page_break.page_numbers == [1]
    assert page_break.alignment_confidence == 1.0
    assert second.page_numbers == [2]


def test_columns_are_scoped_to_their_word_section() -> None:
    payload = _docx(
        """
        <w:p><w:r><w:t>One-column introduction</w:t></w:r></w:p>
        <w:p><w:pPr><w:sectPr><w:cols w:num="1"/></w:sectPr></w:pPr>
          <w:r><w:t>End of the one-column section</w:t></w:r></w:p>
        <w:p><w:r><w:t>Two-column appendix</w:t></w:r></w:p>
        <w:sectPr><w:cols w:num="2"/></w:sectPr>
        """
    )

    blocks, metadata = inspect_docx(payload)

    assert [block.column_count for block in blocks] == [1, 1, 2]
    assert metadata["columns"] == 2


def test_macro_enabled_and_legacy_documents_are_rejected() -> None:
    macro_payload = _docx("", extra_parts={"word/vbaProject.bin": b"macro"})

    with pytest.raises(DocxProcessingError, match="Macro-enabled"):
        inspect_docx(macro_payload)
    with pytest.raises(DocxProcessingError, match="Legacy"):
        inspect_docx(b"\xd0\xcf\x11\xe0" + b"x" * 64)


def test_native_relationships_connect_caption_and_cross_reference() -> None:
    figure = NativeBlock(
        "word/document.xml:block-2",
        "paragraph",
        "Diagram",
        has_drawing=True,
    )
    caption = NativeBlock(
        "word/document.xml:block-3",
        "paragraph",
        "Figure 2 — Request flow",
        style="Caption",
    )
    reference = NativeBlock(
        "word/document.xml:block-4",
        "paragraph",
        "See Fig. 2 for the request flow.",
    )

    relationships = build_native_relationships([figure, caption, reference])

    assert {
        (relationship["type"], relationship["source_id"], relationship["target_id"])
        for relationship in relationships
    } >= {
        ("captioned_by", figure.source_id, caption.source_id),
        ("references", reference.source_id, figure.source_id),
    }

    xml_payload = build_content_tree_xml(
        1,
        [figure, caption, reference],
        "test-session",
        relationships=relationships,
    )
    valid, errors = validate_content_tree(xml_payload, SCHEMA_DIRECTORY)

    assert valid is True
    assert errors == []
    assert b"<caption>Figure 2" in xml_payload
    assert b'cross_reference target="word/document.xml:block-2" type="references"' in xml_payload


def test_page_plan_rasterizes_only_the_visual_page() -> None:
    fitz = pytest.importorskip("fitz")
    payload = _docx(
        """
        <w:p><w:r><w:t>Native only page</w:t></w:r></w:p>
        <w:p><w:r><w:drawing><wp:anchor><wp:docPr descr="Flow diagram"/></wp:anchor></w:drawing></w:r><w:r><w:t>Figure page</w:t></w:r></w:p>
        <w:p><w:pPr><w:pStyle w:val="Caption"/></w:pPr><w:r><w:t>Figure 1 — Flow diagram</w:t></w:r></w:p>
        """
    )
    pdf = fitz.open()
    first = pdf.new_page()
    first.insert_text((72, 72), "Native only page")
    second = pdf.new_page()
    second.insert_text((72, 72), "Figure page")
    # PDF extractors can substitute punctuation during DOCX rendering. The
    # native em dash must still align to this rendered middle dot page.
    second.insert_text((72, 96), "Figure 1 · Flow diagram")
    pdf_payload = pdf.tobytes()
    pdf.close()

    page_plan, images = create_page_plan(
        payload,
        pdf_payload,
        "test-session",
        SCHEMA_DIRECTORY,
    )

    assert [page["route"] for page in page_plan["pages"]] == [
        "native_docx",
        "hybrid",
    ]
    assert set(images) == {2}
    assert page_plan["pages"][0]["content_tree_schema_valid"] is True
    assert page_plan["pages"][0]["native_fidelity"] == "complete"
    assert page_plan["pages"][0]["deterministic_review_required"] is False
    assert any(
        relationship["type"] == "captioned_by"
        for relationship in page_plan["pages"][1]["relationships"]
    )
